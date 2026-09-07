"""统一的 LLM / VLM / Embedding 调用封装

- Embedding: 本地 sentence-transformers（默认 BAAI/bge-small-zh-v1.5），零外部依赖
- Chat / 工具调用 / 多模态: 走 providers.py 的多 provider 适配层

重依赖（sentence-transformers）在函数内 import，保证只做消息编排的模块
（agent / graph / 单元测试）不必安装模型栈即可导入。
"""
from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import llm_spec, load_config, vlm_spec
from . import providers
from .llm_types import LLMResponse, ToolDefinition, user_message


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(load_config()["embedding"]["model"])


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _embedding_model().encode(texts, normalize_embeddings=True).tolist()


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=6))
def complete(
    messages: list[dict],
    *,
    system: str | None = None,
    tools: list[ToolDefinition] | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    multimodal: bool = False,
) -> LLMResponse:
    """按需选择文本 / 多模态模型，调一次模型并返回中立结果。"""
    cfg = load_config()
    spec = _pick_spec(multimodal)
    return providers.complete(
        spec,
        messages,
        system=system,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens or cfg["generation"]["max_tokens"],
    )


def _pick_spec(multimodal: bool):
    if not multimodal:
        return llm_spec()
    spec = vlm_spec()
    if spec is None:
        raise RuntimeError(
            "本次调用需要多模态模型，但 config.yaml 的 `vlm:` 未启用或未配置 API Key。\n"
            "请设置 vlm.enabled=true、填入 vlm.model / vlm.base_url，并在 .env 中提供对应的 key；"
            "或改用纯文本功能（问答 / 任务规划 / 合规校验均不需要视觉模型）。"
        )
    return spec


def chat(
    messages: list[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    """纯文本对话，返回回答文本。messages 支持 role=system 的首条消息。"""
    system_text = None
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            chat_messages.append(m)
    resp = complete(
        chat_messages, system=system_text, temperature=temperature, max_tokens=max_tokens
    )
    return resp.text


def chat_with_image(
    image_path: str | Path,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
) -> str:
    resp = complete(
        [user_message(prompt, image_path=image_path)],
        system=system,
        temperature=temperature,
        multimodal=True,
    )
    return resp.text


def chunked(iterable: Iterable, size: int):
    chunk: list = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
