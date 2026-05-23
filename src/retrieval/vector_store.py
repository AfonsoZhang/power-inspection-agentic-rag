"""Chroma 向量库封装

两套独立 collection：
- regulations: 行业规程片段
- defect_cases: 历史缺陷案例

为避免 Windows 下 Chinese 路径偶发问题，persist_dir 已通过绝对路径传入。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from ..config import load_config


def get_client() -> chromadb.api.ClientAPI:
    cfg = load_config()
    persist_dir: Path = cfg["_paths"]["chroma_dir"]
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_or_create_collection(name: str):
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(name: str):
    client = get_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert(collection, ids: list[str], embeddings: list[list[float]],
           documents: list[str], metadatas: list[dict[str, Any]]) -> None:
    cleaned_metadatas = [_clean_metadata(m) for m in metadatas]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=cleaned_metadatas,
    )


def _clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma 仅接受 str / int / float / bool 字段。"""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = ", ".join(str(x) for x in v)
        else:
            out[k] = str(v)
    return out


def query(collection, embedding: list[float], top_k: int = 5,
          where: dict | None = None) -> list[dict]:
    res = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where,
    )
    out: list[dict] = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for i, doc, meta, dist in zip(ids, docs, metas, dists):
        out.append(
            {
                "id": i,
                "document": doc,
                "metadata": meta or {},
                "distance": float(dist) if dist is not None else None,
                "score": 1.0 - float(dist) if dist is not None else None,
            }
        )
    return out
