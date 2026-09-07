"""Agentic RAG 核心：手写 ReAct 循环

LLM 自主决定调用哪些工具，多轮推理后给出最终回答。
每一步的工具调用和推理过程都被记录，供前端展示 Agent 思考链。

与 graph.py 的关系：这里是**最小可用**的 ReAct 基线（只有 agent↔tools 一个内循环）；
graph.py 在同一套工具上叠加了路由、质检与反思重试，两者可在前端直接对比。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import load_config
from ..generation.llm_client import complete
from ..generation.llm_types import assistant_message, tool_result_message, user_message
from .tools import TOOL_DEFINITIONS, execute_tool

AGENT_SYSTEM = """你是一个电力（无人机）巡检领域的智能助手（Agentic RAG）。

你可以使用以下工具：

检索类——
- search_regulations: 检索行业规程条款
- search_cases: 检索历史缺陷案例
- lookup_asset: 查询资产档案（含经纬度）
- lookup_asset_history: 查询巡检历史

作业类（结果为确定性计算，直接采信，不要自行改算）——
- plan_inspection_mission: 生成复飞任务计划与架次编排
- check_flight_clearance: 起飞前空域/气象/安全距离合规校验

工作流程：
1. 分析用户问题，判断需要哪些信息
2. 主动调用工具获取所需知识（可以多次调用不同工具）
3. 基于工具返回的内容给出有引用的回答

原则：
- 所有结论必须基于工具返回的内容，标注引用来源
- 如果第一次检索结果不够，可以换关键词再搜
- 不要编造规程条款、案例编号、杆塔编号或空域数据
- 涉及放飞决策时，必须先做合规校验再给结论
- 中文回答，结构化输出"""


@dataclass
class AgentStep:
    """Agent 单步记录"""

    step_type: str  # thinking | tool_call | tool_result | router | grade | reflect | answer
    content: str
    tool_name: str | None = None
    tool_input: dict | None = None


@dataclass
class AgentResult:
    """Agent 完整执行结果"""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_turns: int = 0


def max_turns() -> int:
    return int(load_config()["agent"]["max_turns"])


def truncate(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def thinking_steps(resp) -> list[AgentStep]:
    steps = []
    if resp.thinking.strip():
        steps.append(AgentStep(step_type="thinking", content=truncate(resp.thinking)))
    if resp.text.strip():
        steps.append(AgentStep(step_type="thinking", content=resp.text))
    return steps


def run_tool_calls(resp) -> tuple[list[AgentStep], list[dict]]:
    """执行一轮里的全部工具调用，返回 (思考链步骤, 回灌给模型的 tool 消息)。"""
    steps: list[AgentStep] = []
    messages: list[dict] = []
    for call in resp.tool_calls:
        steps.append(AgentStep(
            step_type="tool_call", content=f"调用 {call.name}",
            tool_name=call.name, tool_input=call.arguments,
        ))
        result = execute_tool(call.name, call.arguments)
        steps.append(AgentStep(step_type="tool_result", content=truncate(result), tool_name=call.name))
        messages.append(tool_result_message(call, result))
    return steps, messages


FINAL_TURN_INSTRUCTION = (
    "\n\n[重要] 你已无法再调用任何工具。请直接基于上文已检索到的资料，"
    "用中文给出最终的结构化回答；严禁输出任何工具调用语法。"
)


def run_agent(question: str, image_path: str | None = None) -> AgentResult:
    """执行 Agentic RAG：LLM 自主调用工具，多轮推理后回答。"""
    cfg = load_config()
    limit = max_turns()
    temperature = cfg["agent"]["temperature"]

    steps: list[AgentStep] = []
    messages = [user_message(question, image_path=image_path)]

    for turn in range(limit + 1):
        # 最后一轮撤掉工具定义，强制模型收口出文本答案，避免"到上限了却没有答案"
        use_tools = turn < limit
        system = AGENT_SYSTEM if use_tools else AGENT_SYSTEM + FINAL_TURN_INSTRUCTION

        resp = complete(
            messages,
            system=system,
            tools=TOOL_DEFINITIONS if use_tools else None,
            temperature=temperature,
            max_tokens=4096,
            multimodal=bool(image_path),
        )
        messages.append(assistant_message(resp))
        steps.extend(thinking_steps(resp))

        if not resp.has_tool_calls:
            steps.append(AgentStep(step_type="answer", content=resp.text))
            return AgentResult(answer=resp.text, steps=steps, total_turns=turn + 1)

        tool_steps, tool_messages = run_tool_calls(resp)
        steps.extend(tool_steps)
        messages.extend(tool_messages)

    # 理论不可达：最后一轮已禁用工具，必定走上面的 return
    fallback = "达到最大推理轮次，请尝试更具体的问题。"
    steps.append(AgentStep(step_type="answer", content=fallback))
    return AgentResult(answer=fallback, steps=steps, total_turns=limit + 1)
