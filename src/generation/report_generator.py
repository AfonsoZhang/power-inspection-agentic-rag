"""端到端管线：图像 -> 检测 -> 多路召回 -> VLM/LLM 生成诊断/报告/QA"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import load_config
from ..detection.yolo_stub import detect, detect_to_query
from ..retrieval.retriever import (
    fuse_rrf,
    retrieve_asset_card,
    retrieve_asset_history,
    retrieve_cases,
    retrieve_regulations,
)
from .llm_client import chat
from .prompts import (
    DIAGNOSIS_SYSTEM,
    DIAGNOSIS_USER_TEMPLATE,
    QA_SYSTEM,
    QA_USER_TEMPLATE,
    REPORT_SYSTEM,
    REPORT_USER_TEMPLATE,
)


@dataclass
class DiagnosisResult:
    image_path: str
    detection: dict
    asset_card: dict | None
    asset_history: list[dict]
    regulation_hits: list[dict]
    case_hits: list[dict]
    diagnosis_md: str
    debug: dict[str, Any] = field(default_factory=dict)


def _format_chunks(chunks: list[dict], label: str) -> str:
    if not chunks:
        return f"（无{label}命中）"
    lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {}) or {}
        src = meta.get("source") or meta.get("case_id") or c.get("id")
        section = meta.get("section_path") or meta.get("defect_type") or ""
        score = c.get("fused_score") or c.get("score")
        score_str = f"score={score:.3f}" if isinstance(score, float) else ""
        head = f"[{i}] 来源: {src} | {section} | {score_str}".strip()
        lines.append(f"{head}\n{c.get('document', '')[:600]}")
    return "\n\n".join(lines)


def _format_asset_card(card: dict | None) -> str:
    if not card:
        return "（未提供资产档案）"
    return (
        f"- 资产编号: {card.get('asset_id')}\n"
        f"- 线路: {card.get('line_name')}\n"
        f"- 类型 / 型号: {card.get('asset_type')} / {card.get('model')}\n"
        f"- 投运年份: {card.get('commission_year')}\n"
        f"- 责任人: {card.get('manager')}\n"
        f"- 绝缘子型号: {card.get('insulator_type')}\n"
        f"- 导线型号: {card.get('conductor_type')}\n"
        f"- 备注: {card.get('remarks')}"
    )


def _format_history(history: list[dict]) -> str:
    if not history:
        return "（该资产无历史巡检记录）"
    return "\n".join(
        f"- {h['date']} | {h['method']} | {h['inspector']} | {h.get('summary', '')} | findings={h.get('findings', [])}"
        for h in history
    )


def diagnose_image(image_path: str | Path, asset_id: str | None = None) -> DiagnosisResult:
    cfg = load_config()
    debug = cfg["_env"]["debug"]

    detection = detect(image_path)
    query_text = detect_to_query(detection)

    reg_hits = retrieve_regulations(query_text)
    case_hits = retrieve_cases(query_text)
    asset_card = retrieve_asset_card(asset_id) if asset_id else None
    asset_history = retrieve_asset_history(asset_id) if asset_id else []

    user_prompt = DIAGNOSIS_USER_TEMPLATE.format(
        image_description=detection,
        asset_card=_format_asset_card(asset_card),
        detection_summary=query_text,
        regulation_chunks=_format_chunks(reg_hits, "规程"),
        case_chunks=_format_chunks(case_hits, "案例"),
        asset_history=_format_history(asset_history),
    )

    diagnosis_md = chat(
        [
            {"role": "system", "content": DIAGNOSIS_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg["generation"]["diagnosis_temperature"],
    )

    return DiagnosisResult(
        image_path=str(image_path),
        detection=detection,
        asset_card=asset_card,
        asset_history=asset_history,
        regulation_hits=reg_hits,
        case_hits=case_hits,
        diagnosis_md=diagnosis_md,
        debug={"query": query_text} if debug else {},
    )


def generate_report(inspection_id: str, date: str, method: str,
                    diagnoses: list[DiagnosisResult]) -> str:
    cfg = load_config()
    assets = sorted({d.asset_card["asset_id"] for d in diagnoses if d.asset_card})
    diag_block_parts: list[str] = []
    for i, d in enumerate(diagnoses, 1):
        diag_block_parts.append(
            f"### 诊断 {i}\n图像: {Path(d.image_path).name}\n资产: {d.asset_card.get('asset_id') if d.asset_card else '未知'}\n\n{d.diagnosis_md}"
        )

    user_prompt = REPORT_USER_TEMPLATE.format(
        inspection_id=inspection_id,
        date=date,
        method=method,
        assets=", ".join(assets) or "未指定",
        diagnoses="\n\n---\n\n".join(diag_block_parts),
    )
    return chat(
        [
            {"role": "system", "content": REPORT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg["generation"]["report_temperature"],
        max_tokens=4096,
    )


def answer_question(question: str) -> dict:
    """问答式入口：检索规程 + 案例 + 融合 -> 回答"""
    reg_hits = retrieve_regulations(question)
    case_hits = retrieve_cases(question)
    fused = fuse_rrf([reg_hits, case_hits], top_k=6)

    context = _format_chunks(fused, "上下文")
    user_prompt = QA_USER_TEMPLATE.format(context=context, question=question)
    answer = chat(
        [
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return {"answer": answer, "contexts": fused}
