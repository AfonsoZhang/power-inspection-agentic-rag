"""provider 适配层：消息与工具 schema 的双向翻译（纯函数，不发网络请求）"""
import json
from types import SimpleNamespace

import pytest

from src.agent.tools import TOOL_DEFINITIONS
from src.generation.llm_types import (
    LLMResponse,
    ToolCall,
    assistant_message,
    tool_result_message,
    user_message,
)
from src.generation.providers import (
    ProviderSpec,
    from_anthropic_response,
    from_openai_response,
    to_anthropic_messages,
    to_anthropic_tools,
    to_openai_messages,
    to_openai_tools,
)

CALL = ToolCall(id="call_1", name="search_regulations", arguments={"query": "绝缘子自爆"})


def _conversation():
    """一轮完整的 ReAct 往返：提问 → 助手要调两个工具 → 两条工具结果 → 助手作答。"""
    call2 = ToolCall(id="call_2", name="lookup_asset", arguments={"asset_id": "JN-110-052"})
    return [
        user_message("JN-110-052 的绝缘子自爆怎么处理？"),
        assistant_message(LLMResponse(text="我先查规程和档案", tool_calls=[CALL, call2])),
        tool_result_message(CALL, "规程片段…"),
        tool_result_message(call2, "档案 JSON…"),
        assistant_message(LLMResponse(text="结论如下…")),
    ]


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------

def test_anthropic_merges_consecutive_tool_results_into_one_user_turn():
    msgs = to_anthropic_messages(_conversation())
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "user", "assistant"], roles
    results = msgs[2]["content"]
    assert [b["type"] for b in results] == ["tool_result", "tool_result"]
    assert [b["tool_use_id"] for b in results] == ["call_1", "call_2"]


def test_anthropic_assistant_turn_carries_tool_use_blocks():
    blocks = to_anthropic_messages(_conversation())[1]["content"]
    assert blocks[0]["type"] == "text"
    tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
    assert [b["name"] for b in tool_blocks] == ["search_regulations", "lookup_asset"]
    assert tool_blocks[0]["input"] == CALL.arguments


def test_anthropic_image_part_becomes_base64_source():
    msg = {"role": "user", "content": [
        {"type": "image", "media_type": "image/jpeg", "data": "QUJD"},
        {"type": "text", "text": "这是什么缺陷"},
    ]}
    content = to_anthropic_messages([msg])[0]["content"]
    assert content[0]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}
    assert content[1] == {"type": "text", "text": "这是什么缺陷"}


def test_anthropic_never_emits_an_empty_assistant_turn():
    msgs = to_anthropic_messages([assistant_message(LLMResponse(text=""))])
    assert msgs[0]["content"], "空 assistant 轮会被 API 拒绝"


def test_anthropic_tool_schema_uses_input_schema_key():
    tools = to_anthropic_tools(TOOL_DEFINITIONS)
    assert {"name", "description", "input_schema"} == set(tools[0])
    assert tools[0]["input_schema"] == TOOL_DEFINITIONS[0]["parameters"]


def test_from_anthropic_response_splits_text_thinking_and_tool_use():
    resp = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="thinking", thinking="先查规程"),
            SimpleNamespace(type="text", text="好的"),
            SimpleNamespace(type="tool_use", id="tu_1", name="search_cases", input={"query": "断股"}),
        ],
    )
    out = from_anthropic_response(resp)
    assert out.text == "好的" and out.thinking == "先查规程"
    assert out.has_tool_calls
    assert out.tool_calls[0] == ToolCall(id="tu_1", name="search_cases", arguments={"query": "断股"})


# --------------------------------------------------------------------------
# OpenAI 兼容
# --------------------------------------------------------------------------

def test_openai_keeps_tool_results_as_separate_tool_role_messages():
    msgs = to_openai_messages(_conversation(), system="SYS")
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "tool", "tool", "assistant"]
    assert msgs[3]["tool_call_id"] == "call_1"


def test_openai_tool_call_arguments_are_json_encoded():
    assistant = to_openai_messages(_conversation())[1]
    payload = assistant["tool_calls"][0]
    assert payload["type"] == "function"
    assert json.loads(payload["function"]["arguments"]) == CALL.arguments


def test_openai_image_part_becomes_data_url():
    msg = {"role": "user", "content": [
        {"type": "image", "media_type": "image/png", "data": "QUJD"},
        {"type": "text", "text": "看图"},
    ]}
    content = to_openai_messages([msg])[0]["content"]
    assert content[0]["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_openai_tool_schema_is_wrapped_in_function_envelope():
    tools = to_openai_tools(TOOL_DEFINITIONS)
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["parameters"] == TOOL_DEFINITIONS[0]["parameters"]


def _openai_resp(content, tool_calls=None, reasoning=None, finish="stop"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)])


def test_from_openai_response_parses_tool_calls():
    tc = SimpleNamespace(id="call_9", function=SimpleNamespace(
        name="lookup_asset", arguments='{"asset_id": "QD-110-103"}'))
    out = from_openai_response(_openai_resp(None, [tc], finish="tool_calls"))
    assert out.text == ""
    assert out.tool_calls == [ToolCall(id="call_9", name="lookup_asset",
                                       arguments={"asset_id": "QD-110-103"})]


def test_from_openai_response_keeps_reasoning_content():
    out = from_openai_response(_openai_resp("答案", reasoning="推理过程"))
    assert out.thinking == "推理过程" and out.text == "答案"


def test_malformed_tool_arguments_are_surfaced_not_swallowed():
    tc = SimpleNamespace(id="c", function=SimpleNamespace(name="lookup_asset", arguments="{not json"))
    out = from_openai_response(_openai_resp(None, [tc]))
    assert out.tool_calls[0].arguments == {"__raw__": "{not json"}


# --------------------------------------------------------------------------
# ProviderSpec
# --------------------------------------------------------------------------

@pytest.mark.parametrize("provider,expected_kind", [
    ("deepseek", "openai"), ("openai", "openai"), ("anthropic", "anthropic"),
])
def test_provider_kind_maps_to_protocol_family(provider, expected_kind):
    spec = ProviderSpec(provider=provider, model="m", base_url="u", api_key="k")
    assert spec.kind == expected_kind


def test_reasoner_model_is_flagged_as_tool_incapable():
    assert not ProviderSpec("deepseek", "deepseek-reasoner", "u", "k").supports_tools
    assert ProviderSpec("deepseek", "deepseek-chat", "u", "k").supports_tools
