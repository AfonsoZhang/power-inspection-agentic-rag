"""Agentic RAG 核心：基于 Anthropic tool use 的 ReAct Agent

LLM 自主决定调用哪些检索工具，多轮推理后给出最终回答。
每一步的工具调用和推理过程都被记录，供前端展示 Agent 思考链。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import get_api_key, load_config
from .tools import TOOL_DEFINITIONS, execute_tool

AGENT_SYSTEM = """你是一个电力巡检领域的智能助手（Agentic RAG）。

你可以使用以下工具来检索知识库：
- search_regulations: 检索行业规程条款
- search_cases: 检索历史缺陷案例
- lookup_asset: 查询资产档案
- lookup_asset_history: 查询巡检历史

工作流程：
1. 分析用户问题，判断需要哪些信息
2. 主动调用工具获取所需知识（可以多次调用不同工具）
3. 基于检索到的信息给出有引用的回答

原则：
- 所有结论必须基于工具返回的内容，标注引用来源
- 如果第一次检索结果不够，可以换关键词再搜
- 不要编造规程条款或案例编号
- 中文回答，结构化输出"""

MAX_TURNS = 8


@dataclass
class AgentStep:
    """Agent 单步记录"""
    step_type: str  # "thinking" | "tool_call" | "tool_result" | "answer"
    content: str
    tool_name: str | None = None
    tool_input: dict | None = None


@dataclass
class AgentResult:
    """Agent 完整执行结果"""
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_turns: int = 0


def run_agent(question: str, image_path: str | None = None) -> AgentResult:
    """执行 Agentic RAG：LLM 自主调用工具，多轮推理后回答"""
    from anthropic import Anthropic

    cfg = load_config()
    client = Anthropic(
        api_key=get_api_key(),
        base_url=cfg["_env"]["llm_base_url"],
        timeout=cfg["provider"]["request_timeout"],
    )

    steps: list[AgentStep] = []

    user_content = question
    if image_path:
        import base64
        from pathlib import Path
        path = Path(image_path)
        suffix = path.suffix.lower().lstrip(".")
        media_type = f"image/{'jpeg' if suffix in ('jpg', 'jpeg') else suffix}"
        data = base64.b64encode(path.read_bytes()).decode()
        user_content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            },
            {"type": "text", "text": question},
        ]

    messages = [{"role": "user", "content": user_content}]
    model = cfg["provider"]["vlm_model"] if image_path else cfg["provider"]["llm_model"]

    for turn in range(MAX_TURNS):
        resp = client.messages.create(
            model=model,
            system=AGENT_SYSTEM,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            max_tokens=4096,
            temperature=0.2,
        )

        assistant_content = resp.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_uses = [b for b in assistant_content if b.type == "tool_use"]

        for block in assistant_content:
            if block.type == "thinking":
                steps.append(AgentStep(step_type="thinking", content=block.thinking[:300]))
            elif block.type == "text" and block.text.strip():
                steps.append(AgentStep(step_type="thinking", content=block.text))

        if not tool_uses:
            final_text = ""
            for block in assistant_content:
                if block.type == "text":
                    final_text += block.text
            steps.append(AgentStep(step_type="answer", content=final_text))
            return AgentResult(answer=final_text, steps=steps, total_turns=turn + 1)

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

        messages.append({"role": "user", "content": tool_results})

    final = "达到最大推理轮次，请尝试更具体的问题。"
    steps.append(AgentStep(step_type="answer", content=final))
    return AgentResult(answer=final, steps=steps, total_turns=MAX_TURNS)
