"""Agentic RAG 的 LangGraph 编排版（与 agent.py 的手写 ReAct 循环等价）

把 agent.py 里隐式的 `for turn in range(MAX_TURNS)` 控制流，显式建模成一张
LangGraph StateGraph：

    entry → [agent] ──(还要调工具?)──→ [tools] ──┐
                │                                 │
                └──(无 tool_use / 到上限)→ END     └──→ 回到 [agent]

- agent 节点：调 MiMo（Anthropic 协议）模型，带工具定义，记录思考/判断是否结束。
- tools 节点：执行 tool_use，把 tool_result 回填进 messages。
- 条件边：依据最近一条 assistant 是否含 tool_use（及轮次上限）决定继续还是结束。

模型直接复用现有 Anthropic 客户端调用，不引入 langchain-anthropic / create_react_agent，
以避开 MiMo 推理模型 ThinkingBlock 与 bind_tools 的兼容问题。工具与系统提示词等
资产全部复用 agent.py / tools.py，不重复定义。
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from ..config import get_api_key, load_config
from .agent import AGENT_SYSTEM, MAX_TURNS, AgentResult, AgentStep
from .tools import TOOL_DEFINITIONS, execute_tool


class AgentState(TypedDict):
    """图在各节点间流转的状态。

    messages / steps 用 operator.add 作为 reducer：节点只返回"新增量"，由 LangGraph
    累加到既有列表上，这正是 ReAct 循环里 messages 不断追加的语义。
    """

    client: object                                    # 复用的 Anthropic 客户端
    model: str                                        # 文本或多模态模型名
    messages: Annotated[list, operator.add]           # Anthropic messages 数组
    steps: Annotated[list[AgentStep], operator.add]   # 思考链（供前端展示）
    turn: int                                         # 已完成的 agent 轮次
    answer: str                                       # 最终回答（终止时写入）


def _llm_node(state: AgentState) -> dict:
    """调模型一轮：等价 agent.py:85-111。"""
    resp = state["client"].messages.create(
        model=state["model"],
        system=AGENT_SYSTEM,
        messages=state["messages"],
        tools=TOOL_DEFINITIONS,
        max_tokens=4096,
        temperature=0.2,
    )
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
        final_text = "".join(b.text for b in assistant_content if b.type == "text")
        out["answer"] = final_text
        out["steps"] = steps + [AgentStep(step_type="answer", content=final_text)]
    return out


def _tools_node(state: AgentState) -> dict:
    """执行最近一轮的 tool_use：等价 agent.py:113-136。"""
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

    return {
        "messages": [{"role": "user", "content": tool_results}],
        "steps": steps,
    }


def _should_continue(state: AgentState) -> str:
    """条件边：还要调工具就去 tools，否则（或到轮次上限）结束。"""
    last = state["messages"][-1]
    has_tool = any(b.type == "tool_use" for b in last["content"])
    if not has_tool or state["turn"] >= MAX_TURNS:
        return END
    return "tools"


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", _llm_node)
    g.add_node("tools", _tools_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


_GRAPH = None


def _graph():
    """编译一次、复用。可用 `_graph().get_graph().draw_mermaid()` 导出图结构。"""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


def _build_user_content(question: str, image_path: str | None):
    """构造首条 user 消息内容：等价 agent.py:65-82 的图像 base64 分支。"""
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
    """LangGraph 版 Agentic RAG，与 agent.run_agent 同签名、同返回类型。"""
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
        "messages": [{"role": "user", "content": _build_user_content(question, image_path)}],
        "steps": [],
        "turn": 0,
        "answer": "",
    }

    final = _graph().invoke(init_state, config={"recursion_limit": MAX_TURNS * 2 + 2})

    steps = final["steps"]
    answer = final.get("answer", "")
    if not answer:
        answer = "达到最大推理轮次，请尝试更具体的问题。"
        steps = steps + [AgentStep(step_type="answer", content=answer)]

    return AgentResult(answer=answer, steps=steps, total_turns=final["turn"])
