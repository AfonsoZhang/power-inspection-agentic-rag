"""缺陷检测模块（v1 stub 版本）

v1 不引入 ultralytics 重依赖，转由 VLM 直接给出"视觉描述 + 候选缺陷类型"。
v1.5 计划接入真实 YOLOv8 模型（复用候选人已有 YOLO 项目权重）。

接口约定：
    detect(image_path) -> {
        "device_type": str,
        "candidate_defects": [str, ...],
        "keywords": [str, ...],
        "image_quality": str,
    }
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..generation.llm_client import chat_with_image
from ..generation.prompts import VLM_DEFECT_DESCRIBE_SYSTEM, VLM_DEFECT_DESCRIBE_USER

JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
JSON_BARE_RE = re.compile(r"(\{.*\})", re.DOTALL)


def detect(image_path: str | Path) -> dict:
    raw = chat_with_image(
        image_path,
        VLM_DEFECT_DESCRIBE_USER,
        system=VLM_DEFECT_DESCRIBE_SYSTEM,
        temperature=0.1,
    )
    return _parse_json(raw)


def _parse_json(raw: str) -> dict:
    fallback = {
        "device_type": "未识别",
        "defects": [],
        "keywords": [],
        "image_quality": "未评估",
        "_raw": raw,
    }
    text = raw.strip()
    m = JSON_BLOCK_RE.search(text) or JSON_BARE_RE.search(text)
    if not m:
        return fallback
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return fallback
    fallback.update({k: data.get(k, fallback.get(k)) for k in ("device_type", "defects", "keywords", "image_quality")})
    return fallback


def detect_to_query(detection: dict) -> str:
    """把检测结果拼成可用于检索的 query 字符串"""
    parts: list[str] = []
    if detection.get("device_type"):
        parts.append(f"设备: {detection['device_type']}")
    defects = detection.get("defects") or []
    if defects:
        descs = []
        for d in defects:
            if isinstance(d, dict):
                descs.append(d.get("description") or d.get("type") or str(d))
            else:
                descs.append(str(d))
        parts.append("疑似缺陷: " + "; ".join(descs))
    kws = detection.get("keywords") or []
    if kws:
        parts.append("关键词: " + ", ".join(str(k) for k in kws))
    return "\n".join(parts)
