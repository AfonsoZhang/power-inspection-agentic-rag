"""统一的 LLM / Embedding 调用封装

- Embedding: 本地 sentence-transformers（BAAI/bge-small-zh-v1.5）
- Chat/VLM: MiMo API（Anthropic 协议）
"""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_api_key, load_config


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    cfg = load_config()
    model_name = cfg["provider"]["embedding_model"]
    return SentenceTransformer(model_name)


def _client() -> Anthropic:
    cfg = load_config()
    return Anthropic(
        api_key=get_api_key(),
        base_url=cfg["_env"]["llm_base_url"],
        timeout=cfg["provider"]["request_timeout"],
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=6))
def chat(
    messages: list[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    cfg = load_config()
    client = _client()

    system_text = None
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            chat_messages.append(m)

    kwargs: dict = dict(
        model=cfg["provider"]["llm_model"],
        messages=chat_messages,
        temperature=temperature,
        max_tokens=max_tokens or cfg["generation"]["max_tokens"],
    )
    if system_text:
        kwargs["system"] = system_text

    resp = client.messages.create(**kwargs)
    return _extract_text(resp)


def _extract_text(resp) -> str:
    for block in resp.content:
        if block.type == "text":
            return block.text
    return ""


def _encode_image(image_path: str | Path) -> tuple[str, str]:
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".")
    media_type = f"image/{'jpeg' if suffix in ('jpg', 'jpeg') else suffix}"
    data = base64.b64encode(path.read_bytes()).decode()
    return media_type, data


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=6))
def chat_with_image(
    image_path: str | Path,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
) -> str:
    cfg = load_config()
    client = _client()
    media_type, data = _encode_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    kwargs: dict = dict(
        model=cfg["provider"]["vlm_model"],
        messages=messages,
        temperature=temperature,
        max_tokens=cfg["generation"]["max_tokens"],
    )
    if system:
        kwargs["system"] = system

    resp = client.messages.create(**kwargs)
    return _extract_text(resp)


def chunked(iterable: Iterable, size: int):
    chunk: list = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
