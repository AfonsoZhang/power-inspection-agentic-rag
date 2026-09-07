"""provider 中立的消息 / 工具调用数据结构

Agent 层（agent.py / graph.py）只依赖这里的中立结构，不再直接触碰某一家 SDK 的
content block 对象。各家协议的差异全部收敛在 providers.py 的 adapter 里。

中立消息格式（list[dict]）：
    {"role": "user",      "content": str | list[ContentPart]}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
    {"role": "tool",      "tool_call_id": str, "name": str, "content": str}

ContentPart:
    {"type": "text",  "text": str}
    {"type": "image", "media_type": "image/jpeg", "data": "<base64>"}

system 提示不放进 messages，作为单独参数传给 provider。
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path

# 各 provider 通用的工具定义格式（沿用 JSON Schema 描述入参）
ToolDefinition = dict


@dataclass
class ToolCall:
    """一次工具调用请求。arguments 已解析为 dict。"""

    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """一次模型调用的中立结果。"""

    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def encode_image(image_path: str | Path) -> dict:
    """把本地图片读成中立的 image ContentPart（唯一一处 base64 编码实现）。"""
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".")
    media_type = f"image/{'jpeg' if suffix in ('jpg', 'jpeg') else suffix}"
    return {
        "type": "image",
        "media_type": media_type,
        "data": base64.b64encode(path.read_bytes()).decode(),
    }


def user_message(text: str, image_path: str | Path | None = None) -> dict:
    if not image_path:
        return {"role": "user", "content": text}
    return {
        "role": "user",
        "content": [encode_image(image_path), {"type": "text", "text": text}],
    }


def assistant_message(resp: LLMResponse) -> dict:
    return {"role": "assistant", "content": resp.text, "tool_calls": resp.tool_calls}


def tool_result_message(call: ToolCall, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": content,
    }
