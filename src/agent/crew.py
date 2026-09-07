"""多智能体协作：规划员 → 诊断员 → 调度员

为什么拆角色而不是继续加工具：单个 ReAct Agent 拿到 6 个工具时，"排架次"和"查规程"
两类任务会互相干扰——提示词要同时约束两套输出格式，工具选择也更容易跑偏。
拆开之后每个角色只看得见自己该用的工具子集（tool scoping），提示词各自单一职责，
上一棒的产物作为下一棒的输入写在共享黑板（state）上。

    [planner]  规划员   工具: plan_inspection_mission, lookup_asset
        │              产物: 待飞塔位与架次编排
        ▼
    [clearance] 合规闸门（确定性，不调模型）
        │              产物: 逐基放行结论；全员禁飞则直接跳到调度员出改期建议
        ├── 全部禁飞 ──────────────┐
        ▼                           │
    [diagnostician] 诊断员          │  工具: search_regulations, search_cases, lookup_asset_history
        │              产物: 逐基缺陷定性、处置措施与规程依据
        ▼                           │
    [dispatcher] 调度员 ◄───────────┘  工具: 无（只做综合，避免它再去检索而偏离前两棒的结论）
                       产物: 班组派工单

闸门放在两个模型角色之间、且是确定性的，是这条链的关键：能不能飞是硬约束，
不能让模型"觉得可以"就往下走。
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from ..config import load_config
from ..generation.llm_client import complete
from ..generation.llm_types import assistant_message, user_message
from ..mission.airspace import check_flight, format_report
from .agent import AgentStep, run_tool_calls, thinking_steps
from .tools import TOOL_DEFINITIONS

ROLE_TOOLS = {
    "planner": ("plan_inspection_mission", "lookup_asset"),
    "diagnostician": ("search_regulations", "search_cases", "lookup_asset_history"),
    "dispatcher": (),
}

PLANNER_SYSTEM = """你是电力巡检班组的**任务规划员**。

职责：确定"明天该飞哪些杆塔、怎么排架次"。
- 必须调用 plan_inspection_mission 获取计划，不要自己推算时效或航程
- 输出：① 待飞杆塔编号列表（逐个列出，供后续环节使用）② 架次编排 ③ 排期依据一句话
- 不要给缺陷处置建议，那是诊断员的职责"""

DIAGNOSTICIAN_SYSTEM = """你是电力巡检班组的**缺陷诊断员**。

职责：对规划员选出的杆塔，逐基给出缺陷定性、等级判定与处置措施。
- 结论必须来自 search_regulations / search_cases 的返回内容，标注规程章节号与 case_id
- 证据不足就写"证据不足，需现场复核"，不得编造
- 不要重新排架次，那是规划员的职责"""

DISPATCHER_SYSTEM = """你是电力巡检班组的**调度员**。

职责：把规划员的架次编排、合规闸门的放行结论、诊断员的处置意见，合成一份可直接下发的派工单。
- 结构：作业日期 / 架次与塔位 / 空域与气象约束 / 逐基作业要点与所需备件 / 风险提示
- 只能使用上文已给出的信息，不得新增任何未出现过的杆塔、规程或数值
- 若合规闸门判定禁飞，派工单改为"改期建议 + 替代方案（人工登塔或申请空域许可）\""""


@dataclass
class CrewResult:
    plan_md: str = ""
    clearance_md: str = ""
    diagnosis_md: str = ""
    dispatch_md: str = ""
    asset_ids: list[str] = field(default_factory=list)
    clearance_verdict: str = ""
    steps: list[AgentStep] = field(default_factory=list)


class CrewState(TypedDict):
    line_name: str
    reference_date: str
    weather: dict | None
    agl_m: float | None
    asset_ids: list[str]
    plan_md: str
    clearance_md: str
    clearance_verdict: str
    diagnosis_md: str
    dispatch_md: str
    steps: Annotated[list[AgentStep], operator.add]


def _tools_for(role: str) -> list[dict] | None:
    names = ROLE_TOOLS[role]
    return [t for t in TOOL_DEFINITIONS if t["name"] in names] or None


def _run_role(role: str, system: str, task: str, *, max_turns: int = 4) -> tuple[str, list[AgentStep]]:
    """跑一个角色的小 ReAct 循环，返回 (最终文本, 思考链)。"""
    cfg = load_config()
    tools = _tools_for(role)
    messages = [user_message(task)]
    steps: list[AgentStep] = [AgentStep(step_type="router", content=f"▶ {role} 接棒")]

    for turn in range(max_turns + 1):
        use_tools = bool(tools) and turn < max_turns
        resp = complete(
            messages,
            system=system,
            tools=tools if use_tools else None,
            temperature=cfg["agent"]["temperature"],
            max_tokens=4096,
        )
        messages.append(assistant_message(resp))
        steps.extend(thinking_steps(resp))
        if not resp.has_tool_calls:
            return resp.text, steps
        tool_steps, tool_messages = run_tool_calls(resp)
        steps.extend(tool_steps)
        messages.extend(tool_messages)
    return "", steps


def _planner_node(state: CrewState) -> dict:
    task = (
        f"请为线路「{state['line_name'] or '全部线路'}」生成 {state['reference_date']} 的巡检任务计划。"
        "调用 plan_inspection_mission（line_name 与 reference_date 照此传入），"
        "然后按职责要求输出。"
    )
    text, steps = _run_role("planner", PLANNER_SYSTEM, task)
    # 塔位列表由确定性规划结果给出，不从模型自由文本里正则抠——避免模型漏写或编号写错
    from ..mission.planner import plan_mission

    plan = plan_mission(state["reference_date"], state["line_name"] or None)
    asset_ids = [aid for s in plan.sorties for aid in s.asset_ids]
    return {"plan_md": text, "asset_ids": asset_ids, "steps": steps}


def _clearance_node(state: CrewState) -> dict:
    """确定性合规闸门：不调模型，直接算。"""
    if not state["asset_ids"]:
        return {
            "clearance_md": "无待飞塔位，跳过合规校验。",
            "clearance_verdict": "allowed",
            "steps": [AgentStep(step_type="grade", content="合规闸门：无待飞塔位")],
        }
    report = check_flight(state["asset_ids"], agl_m=state["agl_m"], weather=state["weather"])
    step = AgentStep(
        step_type="grade",
        content=f"合规闸门：{report.verdict}（{len(report.assets)} 基塔位）",
    )
    return {"clearance_md": format_report(report), "clearance_verdict": report.verdict, "steps": [step]}


def _after_clearance(state: CrewState) -> str:
    """全员禁飞就没必要再花模型调用做逐基诊断，直接让调度员出改期方案。"""
    return "dispatcher" if state["clearance_verdict"] == "forbidden" else "diagnostician"


def _diagnostician_node(state: CrewState) -> dict:
    ids = ", ".join(state["asset_ids"][:12]) or "（无）"
    task = (
        f"规划员给出的待飞杆塔：{ids}\n\n"
        f"规划说明：\n{state['plan_md']}\n\n"
        "请对其中因缺陷复查而入选的杆塔，逐基给出缺陷定性、等级判定与处置措施，并标注规程章节与 case_id。"
    )
    text, steps = _run_role("diagnostician", DIAGNOSTICIAN_SYSTEM, task)
    return {"diagnosis_md": text, "steps": steps}


def _dispatcher_node(state: CrewState) -> dict:
    task = (
        f"作业日期：{state['reference_date']}｜线路：{state['line_name'] or '全部线路'}\n\n"
        f"## 规划员产出\n{state['plan_md']}\n\n"
        f"## 合规闸门结论（确定性判定，必须遵守）\n{state['clearance_md']}\n\n"
        f"## 诊断员产出\n{state['diagnosis_md'] or '（本次未执行逐基诊断）'}\n\n"
        "请据此输出派工单。"
    )
    text, steps = _run_role("dispatcher", DISPATCHER_SYSTEM, task)
    return {"dispatch_md": text, "steps": steps}


def build_crew():
    g = StateGraph(CrewState)
    g.add_node("planner", _planner_node)
    g.add_node("clearance", _clearance_node)
    g.add_node("diagnostician", _diagnostician_node)
    g.add_node("dispatcher", _dispatcher_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "clearance")
    g.add_conditional_edges("clearance", _after_clearance,
                            {"diagnostician": "diagnostician", "dispatcher": "dispatcher"})
    g.add_edge("diagnostician", "dispatcher")
    g.add_edge("dispatcher", END)
    return g.compile()


_CREW = None


def crew():
    global _CREW
    if _CREW is None:
        _CREW = build_crew()
    return _CREW


def run_crew(line_name: str | None = None, reference_date: str | None = None,
             weather: dict | None = None, agl_m: float | None = None) -> CrewResult:
    init: CrewState = {
        "line_name": line_name or "",
        "reference_date": reference_date or date.today().isoformat(),
        "weather": weather,
        "agl_m": agl_m,
        "asset_ids": [],
        "plan_md": "",
        "clearance_md": "",
        "clearance_verdict": "",
        "diagnosis_md": "",
        "dispatch_md": "",
        "steps": [],
    }
    final = crew().invoke(init, config={"recursion_limit": 60})
    return CrewResult(
        plan_md=final["plan_md"],
        clearance_md=final["clearance_md"],
        diagnosis_md=final["diagnosis_md"],
        dispatch_md=final["dispatch_md"],
        asset_ids=final["asset_ids"],
        clearance_verdict=final["clearance_verdict"],
        steps=final["steps"],
    )


__all__ = ["CrewResult", "build_crew", "crew", "run_crew"]
