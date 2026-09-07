"""Agentic RAG 的 LangGraph 编排版（纠错式 RAG：路由 + ReAct + 质检反思）

相比 agent.py 的手写 ReAct 循环，这里用 LangGraph StateGraph 把控制流显式建模，
并加入两个手写循环里没有的能力——入口路由 与 质检-反思重试：

    start → [router] ─┬─(确定性场景)→ [direct_lookup] ─┐
                      │                                 ├→ [agent] ⇄ [tools]
                      └─(需自主编排)───────────────────┘        │
                                                                 ▼
                                                              [grade] ──(通过/反思用尽)→ END
                                                                 │
                                                                 └─(不足)→ [reflect] → [agent]

- router：规则识别意图（复用 src/router/intent_router），零模型调用。命中"资产历史/档案"
  与"起飞前合规校验"这类**所需工具确定**的场景时，直接预取，省掉 LLM 试探选工具的来回。
- agent ↔ tools：与 agent.py 等价的 ReAct 内循环。到达 max_turns 后 agent 不再带工具定义，
  强制产出文本答案，保证进入 grade 时状态干净（无悬空 tool_use）。
- grade：LLM-as-Judge 按共享 rubric 质检答案忠实度。
- reflect：质检不足时注入批评意见、回到 agent 重检索，最多 max_reflections 次。

模型调用统一走 generation/llm_client 的 provider 中立封装，图里不出现任何一家 SDK 的对象。
"""
from __future__ import annotations

import json
import operator
from datetime import date
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from ..config import load_config
from ..generation.llm_client import chat, complete
from ..generation.llm_types import assistant_message, user_message
from ..generation.prompts import FAITHFULNESS_PASS_THRESHOLD, FAITHFULNESS_RUBRIC
from ..router.intent_router import (
    detect_intent,
    extract_asset_id,
    extract_asset_ids,
    extract_line_name,
)
from .agent import (
    AGENT_SYSTEM,
    FINAL_TURN_INSTRUCTION,
    AgentResult,
    AgentStep,
    max_turns,
    run_tool_calls,
    thinking_steps,
    truncate,
)
from .tools import TOOL_DEFINITIONS, execute_tool

# 质检判据复用 prompts.FAITHFULNESS_RUBRIC，与离线 eval/ragas_eval 的 Faithfulness 维度同尺；
# 评委按同一 1-5 标准打分，再由 FAITHFULNESS_PASS_THRESHOLD 映射为"通过/反思"的二元门。
GRADE_SYSTEM = "你是电力巡检问答系统的质检员，按统一的忠实度标准给回答打分（与离线评估同尺）。"

GRADE_USER_TEMPLATE = """请判断【回答】是否忠实于【检索上下文】。

""" + FAITHFULNESS_RUBRIC + """

【问题】
{question}

【检索上下文】
{contexts}

【回答】
{answer}

请只输出一个 JSON 对象，格式：{{"score": <1-5>, "reason": "<一句话理由；若打分低于门槛，请指出缺什么、应补充检索什么>"}}"""


class AgentState(TypedDict):
    """图在各节点间流转的状态。

    messages / steps 用 operator.add 作为 reducer：节点只返回"新增量"，由 LangGraph 累加。
    """

    model_is_multimodal: bool
    question: str
    route: str                                        # "direct" | "agent"
    system_hint: str                                  # router 给 agent 注入的路由提示
    prefetched_context: str                           # 确定性快路径预取的资料，注入 system
    prefetch_calls: list                              # router 定下、由 direct_lookup 执行的工具调用
    messages: Annotated[list, operator.add]           # 中立消息数组
    steps: Annotated[list[AgentStep], operator.add]   # 思考链（供前端展示）
    turn: int                                         # 已完成的 agent 轮次
    reflections: int                                  # 已发生的反思重试次数
    pending_tool_calls: list                          # 最近一轮待执行的工具调用
    answer: str
    grade_verdict: str                                # "sufficient" | "insufficient"
    grade_reason: str


# ---------------------------------------------------------------------------
# router：规则分流
# ---------------------------------------------------------------------------

def _router_node(state: AgentState) -> dict:
    q = state["question"]
    intent = detect_intent(q)

    prefetch = _plan_prefetch(q, intent)
    if prefetch:
        label, hint, calls = prefetch
        return {
            "route": "direct",
            "system_hint": hint,
            "prefetch_calls": calls,
            "steps": [AgentStep(step_type="router", content=label)],
        }

    hints = {
        "ask_regulation": "用户在询问规程/标准/处置要求，优先调用 search_regulations，必要时再 search_cases。",
        "plan_mission": "用户在问巡检/复飞排期，优先调用 plan_inspection_mission，再按需补充规程依据。",
    }
    hint = hints.get(intent, "先判断需要规程条款、历史案例还是作业计算，再选择合适的工具。")
    label = f"意图={intent} → ReAct 智能体路径；{hint}"
    return {"route": "agent", "system_hint": hint, "steps": [AgentStep(step_type="router", content=label)]}


def _plan_prefetch(question: str, intent: str) -> tuple[str, str, list[tuple[str, dict]]] | None:
    """判断能否走确定性快路径，返回 (展示标签, 给 agent 的提示, 待确定性执行的工具调用)。"""
    if intent == "ask_history":
        aid = extract_asset_id(question)
        if aid:
            return (
                f"意图=历史查询（{aid}）→ 确定性快路径（直接预取档案+历史，跳过选工具）",
                f"资产 {aid} 的档案与历史已在系统提示中预取给出，无需再调用 lookup_asset / "
                "lookup_asset_history；如需规程/案例再调相应工具。",
                [("lookup_asset", {"asset_id": aid}),
                 ("lookup_asset_history", {"asset_id": aid, "limit": 5})],
            )

    if intent == "flight_clearance":
        ids = extract_asset_ids(question)
        if ids:
            return (
                f"意图=飞行前合规校验（{len(ids)} 基塔位）→ 确定性快路径（直接校验空域/气象/安全距离）",
                "合规校验结果已在系统提示中给出，无需再调用 check_flight_clearance；"
                "请据此向用户说明结论与处置建议。",
                [("check_flight_clearance", {"asset_ids": ids})],
            )

    if intent == "plan_mission":
        line = extract_line_name(question)
        if line:
            return (
                f"意图=任务规划（{line}）→ 确定性快路径（直接生成架次计划）",
                "任务计划已在系统提示中给出，无需再调用 plan_inspection_mission；"
                "请据此向用户解释排期依据。",
                [("plan_inspection_mission", {"line_name": line,
                                              "reference_date": date.today().isoformat()})],
            )
    return None


def _route_decide(state: AgentState) -> str:
    return "direct_lookup" if state["route"] == "direct" else "agent"


def _direct_lookup_node(state: AgentState) -> dict:
    """确定性快路径：不经 LLM 选工具，直接执行 router 定下的工具调用，结果注入 system。"""
    calls = state.get("prefetch_calls") or []
    steps: list[AgentStep] = []
    blocks: list[str] = []
    for name, args in calls:
        steps.append(AgentStep(step_type="tool_call", content=f"（快路径）确定性调用 {name}",
                               tool_name=name, tool_input=args))
        result = execute_tool(name, args)
        steps.append(AgentStep(step_type="tool_result", content=truncate(result), tool_name=name))
        blocks.append(f"## {name} 结果\n{result}")
    return {"prefetched_context": "\n\n".join(blocks), "steps": steps}


# ---------------------------------------------------------------------------
# agent ⇄ tools
# ---------------------------------------------------------------------------

def _system_prompt(state: AgentState, *, final_turn: bool) -> str:
    parts = [AGENT_SYSTEM]
    if state.get("system_hint"):
        parts.append(f"[路由提示] {state['system_hint']}")
    if state.get("prefetched_context"):
        parts.append(f"[已预取资料]\n{state['prefetched_context']}")
    prompt = "\n\n".join(parts)
    return prompt + FINAL_TURN_INSTRUCTION if final_turn else prompt


def _llm_node(state: AgentState) -> dict:
    limit = max_turns()
    use_tools = state["turn"] < limit

    resp = complete(
        state["messages"],
        system=_system_prompt(state, final_turn=not use_tools),
        tools=TOOL_DEFINITIONS if use_tools else None,
        temperature=load_config()["agent"]["temperature"],
        max_tokens=4096,
        multimodal=state["model_is_multimodal"],
    )

    out: dict = {
        "messages": [assistant_message(resp)],
        "steps": thinking_steps(resp),
        "turn": state["turn"] + 1,
        "pending_tool_calls": resp.tool_calls,
    }
    if not resp.has_tool_calls:
        out["answer"] = resp.text
    return out


def _should_continue(state: AgentState) -> str:
    return "tools" if state["pending_tool_calls"] else "grade"


def _tools_node(state: AgentState) -> dict:
    class _Resp:  # run_tool_calls 只用到 .tool_calls
        tool_calls = state["pending_tool_calls"]

    steps, messages = run_tool_calls(_Resp())
    return {"messages": messages, "steps": steps, "pending_tool_calls": []}


# ---------------------------------------------------------------------------
# grade / reflect
# ---------------------------------------------------------------------------

def _collect_contexts(messages: list) -> str:
    """把所有 tool 消息的内容作为质检的"检索资料"。"""
    return "\n\n".join(m["content"] for m in messages if m["role"] == "tool")


def _parse_grade(text: str) -> dict:
    """从评委输出抽出 {"score", "reason"} JSON（与 eval/ragas_eval._parse_judge_response 同格式）。"""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"score": 0, "reason": "评委输出解析失败"}


def _grade_node(state: AgentState) -> dict:
    answer = state.get("answer", "") or "（未产出最终回答）"
    contexts = _collect_contexts(state["messages"]) or state.get("prefetched_context", "")
    contexts = contexts or "（本轮未检索任何资料）"

    verdict_text = chat(
        [
            {"role": "system", "content": GRADE_SYSTEM},
            {"role": "user", "content": GRADE_USER_TEMPLATE.format(
                question=state["question"], contexts=contexts[:3000], answer=answer[:2000],
            )},
        ],
        temperature=0.0,
        max_tokens=2048,
    )

    parsed = _parse_grade(verdict_text)
    try:
        score = int(parsed.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    reason = str(parsed.get("reason", "")).strip() or verdict_text.strip()[:120]

    insufficient = score < FAITHFULNESS_PASS_THRESHOLD
    label = "不足，需反思重试" if insufficient else "通过"
    step = AgentStep(
        step_type="grade",
        content=f"忠实度评分 {score}/5（门槛 {FAITHFULNESS_PASS_THRESHOLD}）→ {label}。{reason}",
    )
    return {
        "answer": answer,
        "grade_verdict": "insufficient" if insufficient else "sufficient",
        "grade_reason": reason,
        "steps": [step],
    }


def _after_grade(state: AgentState) -> str:
    limit = int(load_config()["agent"]["max_reflections"])
    if state["grade_verdict"] == "insufficient" and state["reflections"] < limit:
        return "reflect"
    return END


def _reflect_node(state: AgentState) -> dict:
    n = state["reflections"] + 1
    feedback = (
        f"质检判定上一轮回答【不足】：{state['grade_reason']}。"
        "请据此重新检索（更换关键词或调用其它工具补充资料）后，给出更完整、有据的回答。"
    )
    return {
        "messages": [user_message(feedback)],
        "reflections": n,
        "steps": [AgentStep(step_type="reflect", content=f"第 {n} 次反思重试：{state['grade_reason']}")],
    }


# ---------------------------------------------------------------------------
# 图装配
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", _router_node)
    g.add_node("direct_lookup", _direct_lookup_node)
    g.add_node("agent", _llm_node)
    g.add_node("tools", _tools_node)
    g.add_node("grade", _grade_node)
    g.add_node("reflect", _reflect_node)

    g.set_entry_point("router")
    g.add_conditional_edges("router", _route_decide,
                            {"direct_lookup": "direct_lookup", "agent": "agent"})
    g.add_edge("direct_lookup", "agent")
    g.add_conditional_edges("agent", _should_continue, {"tools": "tools", "grade": "grade"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("grade", _after_grade, {"reflect": "reflect", END: END})
    g.add_edge("reflect", "agent")
    return g.compile()


_GRAPH = None


def graph():
    """编译一次、复用。可用 `graph().get_graph().draw_mermaid()` 导出图结构。"""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_graph_agent(question: str, image_path: str | None = None) -> AgentResult:
    """LangGraph 版 Agentic RAG（路由 + ReAct + 质检反思），与 agent.run_agent 同签名同返回。"""
    init_state: AgentState = {
        "model_is_multimodal": bool(image_path),
        "question": question,
        "route": "agent",
        "system_hint": "",
        "prefetched_context": "",
        "prefetch_calls": [],
        "messages": [user_message(question, image_path=image_path)],
        "steps": [],
        "turn": 0,
        "reflections": 0,
        "pending_tool_calls": [],
        "answer": "",
        "grade_verdict": "",
        "grade_reason": "",
    }

    final = graph().invoke(init_state, config={"recursion_limit": 80})
    answer = final.get("answer", "") or "达到最大推理轮次，请尝试更具体的问题。"
    steps = final["steps"] + [AgentStep(step_type="answer", content=answer)]
    return AgentResult(answer=answer, steps=steps, total_turns=final["turn"])
