"""轻量意图识别（规则版，零模型调用）

只用来给 LangGraph 的入口路由做**廉价**分流：命中确定性场景就走快路径，
其余交回 ReAct 让模型自己选工具。规则误判的代价只是多绕一次 ReAct，不会给出错误答案，
所以这里刻意不引入 LLM 分类器——那会给每个问题都加一次模型调用。

意图：
- ask_history:      询问某资产历史 / 档案（有资产编号时可确定性预取）
- plan_mission:     排复飞任务 / 架次计划
- flight_clearance: 起飞前空域、气象、能不能飞
- ask_regulation:   询问规程条款、分级、时效
- ask_general:      通用知识问答（兜底）
"""
from __future__ import annotations

import re

ASSET_ID_RE = re.compile(r"[A-Z]{2}-\d{3}-\d{3}")
LINE_NAME_RE = re.compile(r"[一-龥]{2,6}\s*\d{2,3}kV\s*[一-龥]*线路")

HISTORY_KEYWORDS = ("历史", "上次", "之前", "曾经", "以往", "档案")
PLAN_KEYWORDS = ("复飞", "架次", "任务计划", "巡检计划", "排期", "排班", "怎么排", "航线", "作业计划")
CLEARANCE_KEYWORDS = ("能不能飞", "能否飞", "可以飞", "空域", "禁飞", "限高", "净空", "合规校验",
                      "起飞前", "适航", "风速", "能见度")
REGULATION_KEYWORDS = ("规程", "标准", "时效", "处置", "如何处理", "怎么办", "时限",
                       "等级", "分级", "几级", "判定", "定级")


def detect_intent(question: str) -> str:
    if any(k in question for k in CLEARANCE_KEYWORDS):
        return "flight_clearance"
    if any(k in question for k in PLAN_KEYWORDS):
        return "plan_mission"
    if ASSET_ID_RE.search(question) and any(k in question for k in HISTORY_KEYWORDS):
        return "ask_history"
    if any(k in question for k in REGULATION_KEYWORDS):
        return "ask_regulation"
    return "ask_general"


def extract_asset_id(question: str) -> str | None:
    m = ASSET_ID_RE.search(question)
    return m.group(0) if m else None


def extract_asset_ids(question: str) -> list[str]:
    return list(dict.fromkeys(ASSET_ID_RE.findall(question)))


def extract_line_name(question: str) -> str | None:
    m = LINE_NAME_RE.search(question)
    return m.group(0).strip() if m else None
