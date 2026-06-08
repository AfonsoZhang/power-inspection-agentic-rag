"""Agentic RAG 的 LangGraph 编排版（纠错式 RAG：路由 + ReAct + 质检反思）

相比 agent.py 的手写 ReAct 循环，这里用 LangGraph StateGraph 把控制流显式建模，
并加入两个手写循环里没有的能力——入口路由 与 质检-反思重试：

    start → [router] → [agent] ──(assistant 含 tool_use)──→ [tools] ──┐
                          │                                            │
                          │ (无 tool_use / 到轮次上限)                  └─→ 回到 [agent]
                          ▼
                       [grade] ──(充分 / 反思已达上限)──→ END
                          │
                          └──(不足)──→ [reflect] ──→ 回到 [agent]

- router：规则识别意图（复用 src/router/intent_router），给 agent 注入「优先调哪些工具」的提示，
  不额外调模型，零成本。
- agent ↔ tools：与 agent.py 等价的 ReAct 内循环。到达 MAX_TURNS 后，agent 不再带工具定义，
  强制产出文本答案，保证进入 grade 时状态干净（无悬空 tool_use）。
- grade：LLM-as-Judge 质检答案是否「基于检索资料且充分」（复用 llm_client.chat，max_tokens=2048）。
- reflect：质检不足时注入批评意见、回到 agent 重检索，最多 MAX_REFLECTIONS 次。

模型调用全部走原生 Anthropic 客户端 / 现有 chat 封装，不引入 langchain-anthropic /
create_react_agent，规避 MiMo 推理模型 ThinkingBlock 与 bind_tools 的兼容问题。
"""
from __future__ import annotations

import json
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from ..config import get_api_key, load_config
from ..generation.llm_client import chat
from ..generation.prompts import FAITHFULNESS_PASS_THRESHOLD, FAITHFULNESS_RUBRIC
from ..router.intent_router import detect_intent, extract_asset_id
from .agent import AGENT_SYSTEM, MAX_TURNS, AgentResult, AgentStep
from .tools import TOOL_DEFINITIONS, execute_tool

MAX_REFLECTIONS = 2

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

    client: object                                    # 复用的 Anthropic 客户端
    model: str                                        # 文本或多模态模型名
    question: str                                     # 原始问题（供 router / grade 使用）
    system_hint: str                                  # router 给 agent 注入的路由提示
    messages: Annotated[list, operator.add]           # Anthropic messages 数组
    steps: Annotated[list[AgentStep], operator.add]   # 思考链（供前端展示）
    turn: int                                         # 已完成的 agent 轮次
    reflections: int                                  # 已发生的反思重试次数
    answer: str                                       # 最终回答
    grade_verdict: str                                # "sufficient" | "insufficient"
    grade_reason: str                                 # 质检理由


def _router_node(state: AgentState) -> dict:
    """入口路由：规则识别意图，给 agent 注入工具偏好提示（复用 intent_router）。"""
    q = state["question"]
    intent = detect_intent(q)
    asset_id = extract_asset_id(q)

    if intent == "ask_history":
        target = f"资产 {asset_id} " if asset_id else ""
        hint = f"用户在询问{target}的历史情况，优先调用 lookup_asset 与 lookup_asset_history。"
        label = f"意图=历史查询{('（'+asset_id+'）') if asset_id else ''}"
    elif intent == "ask_regulation":
        hint = "用户在询问规程/标准/处置要求，优先调用 search_regulations，必要时再 search_cases。"
        label = "意图=规程查询"
    else:
        hint = "先判断需要规程条款还是历史案例，再选择合适的工具检索。"
        label = "意图=通用问答"

    step = AgentStep(step_type="router", content=f"{label} → {hint}")
    return {"system_hint": hint, "steps": [step]}


def _system_prompt(state: AgentState) -> str:
    hint = state.get("system_hint", "")
    return AGENT_SYSTEM + (f"\n\n[路由提示] {hint}" if hint else "")


def _llm_node(state: AgentState) -> dict:
    """调模型一轮。到达 MAX_TURNS 后不再带工具，强制产出文本答案（保证 grade 状态干净）。"""
    use_tools = state["turn"] < MAX_TURNS

    system = _system_prompt(state)
    if not use_tools:
        # 到达轮次上限：撤掉工具定义强制收尾。必须明确告知模型，否则它可能把工具调用
        # 当成纯文本输出（如 <tool_call><function=...>），污染最终回答。
        system += (
            "\n\n[重要] 你已无法再调用任何工具。请直接基于上文已检索到的资料，"
            "用中文给出最终的结构化回答；严禁输出任何工具调用语法（如 <tool_call> / <function=...>）。"
        )

    kwargs: dict = dict(
        model=state["model"],
        system=system,
        messages=state["messages"],
        max_tokens=4096,
        temperature=0.2,
    )
    if use_tools:
        kwargs["tools"] = TOOL_DEFINITIONS

    resp = state["client"].messages.create(**kwargs)
    assistant_content = resp.content

    steps: list[AgentStep] = []
    for block in assistant_content:
        if block.type == "thinking":
            steps.append(AgentStep(step_type="thinking", content=block.thinking[:300]))
        elif block.type == "text" and block.text.strip():
            steps.append(AgentStep(step_type="thinking", content=block.text))

    out: dict = {
        "messages": [{"role": "assistant", "content": assistant_content}],
        "steps": steps,
        "turn": state["turn"] + 1,
    }

    tool_uses = [b for b in assistant_content if b.type == "tool_use"]
    if not tool_uses:
        out["answer"] = "".join(b.text for b in assistant_content if b.type == "text")
    return out


def _should_continue(state: AgentState) -> str:
    """agent 之后：还要调工具就去 tools，否则去 grade 质检。"""
    last = state["messages"][-1]
    has_tool = any(b.type == "tool_use" for b in last["content"])
    return "tools" if has_tool else "grade"


def _tools_node(state: AgentState) -> dict:
    """执行最近一轮的 tool_use，回填 tool_result。"""
    last = state["messages"][-1]
    tool_uses = [b for b in last["content"] if b.type == "tool_use"]

    steps: list[AgentStep] = []
    tool_results = []
    for tu in tool_uses:
        steps.append(AgentStep(
            step_type="tool_call",
            content=f"调用 {tu.name}",
            tool_name=tu.name,
            tool_input=tu.input,
        ))

        result_text = execute_tool(tu.name, tu.input)

        steps.append(AgentStep(
            step_type="tool_result",
            content=result_text[:300] + "..." if len(result_text) > 300 else result_text,
            tool_name=tu.name,
        ))
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tu.id,
            "content": result_text,
        })

    return {"messages": [{"role": "user", "content": tool_results}], "steps": steps}


def _collect_contexts(messages: list) -> str:
    """从 messages 里抽出所有 tool_result 文本，作为质检的"检索资料"。"""
    parts = []
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            for blk in m["content"]:
                if isinstance(blk, dict) and blk.get("type") == "tool_result":
                    parts.append(str(blk.get("content", "")))
    return "\n\n".join(parts)


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
    """LLM-as-Judge 质检：按共享 rubric 给忠实度打 1-5 分，低于门槛则触发反思。"""
    answer = state.get("answer", "") or "（未产出最终回答）"
    contexts = _collect_contexts(state["messages"]) or "（本轮未检索任何资料）"

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
    verdict = "insufficient" if insufficient else "sufficient"

    label = "不足，需反思重试" if insufficient else "通过"
    step = AgentStep(
        step_type="grade",
        content=f"忠实度评分 {score}/5（门槛 {FAITHFULNESS_PASS_THRESHOLD}）→ {label}。{reason}",
    )
    return {"answer": answer, "grade_verdict": verdict, "grade_reason": reason, "steps": [step]}


def _after_grade(state: AgentState) -> str:
    """grade 之后：不足且仍有重试额度则 reflect，否则结束。"""
    if state["grade_verdict"] == "insufficient" and state["reflections"] < MAX_REFLECTIONS:
        return "reflect"
    return END


def _reflect_node(state: AgentState) -> dict:
    """反思：把质检意见作为反馈注入对话，回到 agent 重检索。"""
    n = state["reflections"] + 1
    feedback = (
        f"质检判定上一轮回答【不足】：{state['grade_reason']}。"
        "请据此重新检索（更换关键词或调用其它工具补充资料）后，给出更完整、有据的回答。"
    )
    step = AgentStep(step_type="reflect", content=f"第 {n} 次反思重试：{state['grade_reason']}")
    return {
        "messages": [{"role": "user", "content": feedback}],
        "reflections": n,
        "steps": [step],
    }


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", _router_node)
    g.add_node("agent", _llm_node)
    g.add_node("tools", _tools_node)
    g.add_node("grade", _grade_node)
    g.add_node("reflect", _reflect_node)

    g.set_entry_point("router")
    g.add_edge("router", "agent")
    g.add_conditional_edges("agent", _should_continue, {"tools": "tools", "grade": "grade"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("grade", _after_grade, {"reflect": "reflect", END: END})
    g.add_edge("reflect", "agent")
    return g.compile()


_GRAPH = None


def _graph():
    """编译一次、复用。可用 `_graph().get_graph().draw_mermaid()` 导出图结构。"""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


def _build_user_content(question: str, image_path: str | None):
    """构造首条 user 消息内容：含图像 base64 分支（等价 agent.py 的图像处理）。"""
    if not image_path:
        return question

    import base64
    from pathlib import Path

    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".")
    media_type = f"image/{'jpeg' if suffix in ('jpg', 'jpeg') else suffix}"
    data = base64.b64encode(path.read_bytes()).decode()
    return [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        },
        {"type": "text", "text": question},
    ]


def run_graph_agent(question: str, image_path: str | None = None) -> AgentResult:
    """LangGraph 版 Agentic RAG（路由 + ReAct + 质检反思），与 agent.run_agent 同签名同返回。"""
    from anthropic import Anthropic

    cfg = load_config()
    client = Anthropic(
        api_key=get_api_key(),
        base_url=cfg["_env"]["llm_base_url"],
        timeout=cfg["provider"]["request_timeout"],
    )
    model = cfg["provider"]["vlm_model"] if image_path else cfg["provider"]["llm_model"]

    init_state: AgentState = {
        "client": client,
        "model": model,
        "question": question,
        "system_hint": "",
        "messages": [{"role": "user", "content": _build_user_content(question, image_path)}],
        "steps": [],
        "turn": 0,
        "reflections": 0,
        "answer": "",
        "grade_verdict": "",
        "grade_reason": "",
    }

    final = _graph().invoke(init_state, config={"recursion_limit": 80})

    steps = final["steps"]
    answer = final.get("answer", "") or "达到最大推理轮次，请尝试更具体的问题。"
    steps = steps + [AgentStep(step_type="answer", content=answer)]

    return AgentResult(answer=answer, steps=steps, total_turns=final["turn"])
