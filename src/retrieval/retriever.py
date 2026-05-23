"""多路召回入口

- retrieve_regulations: 规程文本召回
- retrieve_cases: 历史案例召回（支持 asset_type / defect_type 元数据过滤）
- retrieve_asset_history: 该资产的巡检历史（结构化检索，无需向量化）
- fuse_rrf: 多路结果 RRF 融合
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..config import load_config
from ..generation.llm_client import embed_text
from ..ingestion.text_loader import get_asset_history, load_assets
from .vector_store import get_or_create_collection, query


def retrieve_regulations(question: str, top_k: int | None = None) -> list[dict]:
    cfg = load_config()
    top_k = top_k or cfg["retrieval"]["text_top_k"]
    coll = get_or_create_collection(cfg["vector_store"]["regulations_collection"])
    emb = embed_text(question)
    return query(coll, emb, top_k=top_k)


def retrieve_cases(question: str, *, defect_type: str | None = None,
                   asset_type: str | None = None,
                   top_k: int | None = None) -> list[dict]:
    cfg = load_config()
    top_k = top_k or cfg["retrieval"]["case_top_k"]
    coll = get_or_create_collection(cfg["vector_store"]["cases_collection"])
    emb = embed_text(question)

    where: dict[str, Any] | None = None
    filters: list[dict] = []
    if defect_type:
        filters.append({"defect_type": {"$eq": defect_type}})
    if asset_type:
        filters.append({"asset_type": {"$eq": asset_type}})
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    return query(coll, emb, top_k=top_k, where=where)


def retrieve_asset_card(asset_id: str) -> dict | None:
    return load_assets().get(asset_id)


def retrieve_asset_history(asset_id: str, limit: int = 5) -> list[dict]:
    history = get_asset_history(asset_id)
    history.sort(key=lambda h: h["date"], reverse=True)
    return history[:limit]


def fuse_rrf(result_lists: list[list[dict]], k: int = 60,
             top_k: int = 6) -> list[dict]:
    """Reciprocal Rank Fusion: 不同路召回得分量纲不同，按排名融合更稳健"""
    score: dict[str, float] = defaultdict(float)
    seen: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item["id"]
            score[cid] += 1.0 / (k + rank + 1)
            seen.setdefault(cid, item)
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    out: list[dict] = []
    for cid, s in ranked[:top_k]:
        item = dict(seen[cid])
        item["fused_score"] = s
        out.append(item)
    return out
