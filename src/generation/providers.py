"""多 provider 适配层：把中立消息/工具格式翻译成各家协议，再把回包翻译回中立结构。

支持三类 provider：
- ``openai``   : 任何 OpenAI 兼容端点（DeepSeek / 通义 / vLLM / Ollama / OpenAI 本身）
- ``deepseek`` : ``openai`` 的一个预设（base_url 默认 https://api.deepseek.com）
- ``anthropic``: Anthropic Messages 协议（原 MiMo 走的就是这条）

设计要点：
1. SDK 一律**懒加载**——只装了 openai 的环境不会因为 import anthropic 而崩，
   单元测试也能在不装任何 SDK 的情况下测消息翻译逻辑。
2. 翻译是纯函数（``to_*`` / ``from_*``），不发网络请求，可直接单测。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm_types import LLMResponse, ToolCall

OPENAI_COMPATIBLE = ("openai", "deepseek")

DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}

# 已知不支持 function calling 的模型：走 Agent 路径时需提前给出可读报错，
# 而不是让上游返回一个含义模糊的 400。
NO_TOOL_MODELS = ("deepseek-reasoner",)


@dataclass(frozen=True)
class ProviderSpec:
    """一路模型的完整配置（文本模型和多模态模型各持一份）。"""

    provider: str
    model: str
    base_url: str
    api_key: str
    timeout: int = 120
    max_retries: int = 3

    @property
    def kind(self) -> str:
        """归一到实际使用的协议族。"""
        return "anthropic" if self.provider == "anthropic" else "openai"

    @property
    def supports_tools(self) -> bool:
        return not any(self.model.startswith(m) for m in NO_TOOL_MODELS)


# --------------------------------------------------------------------------
# 消息翻译：中立 -> Anthropic
# --------------------------------------------------------------------------

def _to_anthropic_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if p["type"] == "image":
            parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": p["media_type"],
                    "data": p["data"],
                },
            })
        else:
            parts.append({"type": "text", "text": p["text"]})
    return parts


def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """中立消息 -> Anthropic messages。

    Anthropic 要求同一轮的多个 tool_result 合并进**一条** user 消息，
    所以连续的 role="tool" 消息在这里被折叠。
    """
    out: list[dict] = []
    pending_results: list[dict] = []

    def flush_results():
        if pending_results:
            out.append({"role": "user", "content": pending_results[:]})
            pending_results.clear()

    for m in messages:
        role = m["role"]
        if role == "tool":
            pending_results.append({
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m["content"],
            })
            continue

        flush_results()
        if role == "assistant":
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls") or []:
                blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            # 空 assistant 轮会被 API 拒绝，补一个占位文本
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "..."}]})
        else:
            out.append({"role": "user", "content": _to_anthropic_content(m["content"])})

    flush_results()
    return out


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
        for t in tools
    ]


def from_anthropic_response(resp: Any) -> LLMResponse:
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in resp.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {})))
    return LLMResponse(
        text="".join(text_parts),
        thinking="".join(thinking_parts),
        tool_calls=tool_calls,
        finish_reason=getattr(resp, "stop_reason", "") or "",
    )


# --------------------------------------------------------------------------
# 消息翻译：中立 -> OpenAI 兼容
# --------------------------------------------------------------------------

def _to_openai_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if p["type"] == "image":
            url = f"data:{p['media_type']};base64,{p['data']}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            parts.append({"type": "text", "text": p["text"]})
    return parts


def to_openai_messages(messages: list[dict], system: str | None = None) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        role = m["role"]
        if role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m["tool_call_id"],
                "content": m["content"],
            })
        elif role == "assistant":
            msg: dict = {"role": "assistant", "content": m.get("content") or ""}
            calls = m.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in calls
                ]
            out.append(msg)
        else:
            out.append({"role": "user", "content": _to_openai_content(m["content"])})
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def from_openai_response(resp: Any) -> LLMResponse:
    choice = resp.choices[0]
    msg = choice.message
    tool_calls: list[ToolCall] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        raw_args = tc.function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            # 模型偶发吐出非法 JSON：不吞掉，交给上层当成工具执行失败处理
            args = {"__raw__": raw_args}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    return LLMResponse(
        text=msg.content or "",
        # DeepSeek reasoner 走 reasoning_content 字段返回思维链
        thinking=getattr(msg, "reasoning_content", "") or "",
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason or "",
    )


# --------------------------------------------------------------------------
# 统一调用入口
# --------------------------------------------------------------------------

_CLIENT_CACHE: dict[tuple, Any] = {}


def _client(spec: ProviderSpec):
    key = (spec.kind, spec.base_url, spec.api_key, spec.timeout)
    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]

    if spec.kind == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=spec.api_key, base_url=spec.base_url, timeout=spec.timeout)
    else:
        from openai import OpenAI

        client = OpenAI(api_key=spec.api_key, base_url=spec.base_url, timeout=spec.timeout)

    _CLIENT_CACHE[key] = client
    return client


def complete(
    spec: ProviderSpec,
    messages: list[dict],
    *,
    system: str | None = None,
    tools: list[dict] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> LLMResponse:
    """按 spec 调一次模型，返回中立结果。"""
    if tools and not spec.supports_tools:
        raise RuntimeError(
            f"模型 {spec.model} 不支持 function calling，无法跑 Agent 工具编排。"
            f"请在 config.yaml 的 llm.model 换成支持工具调用的模型（如 deepseek-chat）。"
        )

    client = _client(spec)

    if spec.kind == "anthropic":
        kwargs: dict = {
            "model": spec.model,
            "messages": to_anthropic_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        return from_anthropic_response(client.messages.create(**kwargs))

    kwargs = {
        "model": spec.model,
        "messages": to_openai_messages(messages, system),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = to_openai_tools(tools)
    return from_openai_response(client.chat.completions.create(**kwargs))
