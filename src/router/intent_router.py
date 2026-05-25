"""轻量意图识别（v1 规则版，v2 计划替换为 LLM 分类器）

意图：
- diagnose:    上传图像后做缺陷诊断（由 UI 直接触发，无需路由）
- ask_history: 询问某资产历史
- ask_regulation: 询问规程条款
- ask_general: 通用知识问答（兜底）
"""
from __future__ import annotations

import re

ASSET_ID_RE = re.compile(r"[A-Z]{2}-\d{3}-\d{3}")
HISTORY_KEYWORDS = ("历史", "上次", "之前", "曾经", "以往")
REGULATION_KEYWORDS = ("规程", "标准", "时效", "处置", "如何处理", "怎么办", "时限", "等级", "判定")


def detect_intent(question: str) -> str:
    if ASSET_ID_RE.search(question) and any(k in question for k in HISTORY_KEYWORDS):
        return "ask_history"
    if any(k in question for k in REGULATION_KEYWORDS):
        return "ask_regulation"
    return "ask_general"


def extract_asset_id(question: str) -> str | None:
    m = ASSET_ID_RE.search(question)
    return m.group(0) if m else None
