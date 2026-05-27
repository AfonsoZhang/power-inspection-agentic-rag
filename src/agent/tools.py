"""Agentic RAG 工具定义（Anthropic tool use 格式）

Agent 可自主选择调用以下工具完成巡检知识问答：
- search_regulations: 检索行业规程
- search_cases: 检索历史缺陷案例
- lookup_asset: 查询资产档案
- lookup_asset_history: 查询资产巡检历史
"""
from __future__ import annotations

from ..retrieval.retriever import (
    fuse_rrf,
    retrieve_asset_card,
    retrieve_asset_history,
    retrieve_cases,
    retrieve_regulations,
)

TOOL_DEFINITIONS = [
    {
        "name": "search_regulations",
        "description": "检索电力巡检行业规程（绝缘子、杆塔、导线金具等），返回相关条款片段。用于回答处置标准、缺陷分级、时效要求等问题。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词，例如'绝缘子自爆处置'、'导线断股分级标准'",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_cases",
        "description": "检索历史缺陷案例库，返回相似的处置案例。可按缺陷类型或资产类型过滤。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "案例检索关键词，例如'复合绝缘子伞裙撕裂'",
                },
                "defect_type": {
                    "type": "string",
                    "description": "可选，按缺陷类型过滤，例如'螺栓松动'、'绝缘子单片自爆'",
                },
                "asset_type": {
                    "type": "string",
                    "description": "可选，按资产类型过滤，例如'110kV角钢塔'、'220kV角钢塔'",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "lookup_asset",
        "description": "根据资产编号查询设备档案卡，包含线路名称、型号、投运年份、责任人等信息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "asset_id": {
                    "type": "string",
                    "description": "资产编号，格式如 JN-110-052、QD-110-103",
                },
            },
            "required": ["asset_id"],
        },
    },
    {
        "name": "lookup_asset_history",
        "description": "查询某资产的历史巡检记录，了解该设备过去的巡检情况和发现的问题。",
        "input_schema": {
            "type": "object",
            "properties": {
                "asset_id": {
                    "type": "string",
                    "description": "资产编号",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回最近几条记录，默认5",
                },
            },
            "required": ["asset_id"],
        },
    },
]


def execute_tool(name: str, args: dict) -> str:
    """执行工具调用，返回结果文本"""
    if name == "search_regulations":
        hits = retrieve_regulations(args["query"], top_k=6)
        if not hits:
            return "未找到相关规程条款。"
        parts = []
        for i, h in enumerate(hits, 1):
            meta = h.get("metadata", {})
            src = meta.get("source", "")
            sec = meta.get("section_path", "")
            score = h.get("score", 0)
            parts.append(f"[{i}] {src} | {sec} | 相似度={score:.3f}\n{h['document'][:500]}")
        return "\n\n".join(parts)

    elif name == "search_cases":
        hits = retrieve_cases(
            args["query"],
            defect_type=args.get("defect_type"),
            asset_type=args.get("asset_type"),
            top_k=5,
        )
        if not hits:
            return "未找到相关历史案例。"
        parts = []
        for i, h in enumerate(hits, 1):
            meta = h.get("metadata", {})
            score = h.get("score", 0)
            parts.append(f"[{i}] {meta.get('case_id', '')} | {meta.get('defect_type', '')} | 相似度={score:.3f}\n{h['document'][:500]}")
        return "\n\n".join(parts)

    elif name == "lookup_asset":
        card = retrieve_asset_card(args["asset_id"])
        if not card:
            return f"未找到资产 {args['asset_id']} 的档案信息。"
        import json
        return json.dumps(card, ensure_ascii=False, indent=2)

    elif name == "lookup_asset_history":
        history = retrieve_asset_history(args["asset_id"], limit=args.get("limit", 5))
        if not history:
            return f"资产 {args['asset_id']} 无历史巡检记录。"
        import json
        return json.dumps(history, ensure_ascii=False, indent=2)

    else:
        return f"未知工具: {name}"
