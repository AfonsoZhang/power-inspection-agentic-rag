"""Streamlit Demo 前端

五个 Tab：
1. 智能问答（Agent） - Agentic RAG，LLM 自主调用工具
2. 缺陷诊断（Agent） - 上传图像，Agent 自主检索 + 诊断
3. 巡检报告 - 多张图像聚合生成报告草稿
4. 知识问答（基础RAG） - 传统 RAG 对比
5. 系统信息 - 配置 / 索引状态
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.agent.agent import run_agent
from src.agent.graph import run_graph_agent
from src.config import load_config
from src.generation.report_generator import (
    answer_question,
    diagnose_image,
    generate_report,
)
from src.ingestion.text_loader import (
    load_assets,
    load_defect_cases,
    load_inspection_history,
    load_regulation_chunks,
)


st.set_page_config(
    page_title="无人机巡检 Agentic RAG",
    page_icon=":mag:",
    layout="wide",
)

TOOL_ICONS = {
    "search_regulations": ":book:",
    "search_cases": ":file_folder:",
    "lookup_asset": ":wrench:",
    "lookup_asset_history": ":clock3:",
}


@st.cache_data(show_spinner=False)
def _assets() -> dict:
    return load_assets()


def _save_upload(file) -> Path:
    suffix = Path(file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir())
    tmp.write(file.getvalue())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _render_agent_steps(steps):
    """渲染 Agent 思考链"""
    for step in steps:
        if step.step_type == "router":
            st.markdown(f"> :compass: **路由:** {step.content}")
        elif step.step_type == "grade":
            st.markdown(f"> :test_tube: **质检:** {step.content}")
        elif step.step_type == "reflect":
            st.markdown(f"> :recycle: **反思重试:** {step.content}")
        elif step.step_type == "thinking":
            st.markdown(f"> :brain: **Agent 思考:** {step.content[:200]}")
        elif step.step_type == "tool_call":
            icon = TOOL_ICONS.get(step.tool_name, ":gear:")
            args_str = json.dumps(step.tool_input, ensure_ascii=False) if step.tool_input else ""
            with st.expander(f"{icon} 调用工具: **{step.tool_name}**({args_str})", expanded=False):
                if step.tool_input:
                    st.json(step.tool_input)
        elif step.step_type == "tool_result":
            with st.expander(f":white_check_mark: {step.tool_name} 返回结果", expanded=False):
                st.text(step.content)
        elif step.step_type == "answer":
            pass


def tab_agent_qa():
    st.header("智能问答（Agentic RAG）")
    st.caption("Agent 自主决定调用哪些工具检索知识库，支持多轮推理。可上传巡检图像辅助诊断。")

    col1, col2 = st.columns([3, 1])
    with col1:
        sample_qs = [
            "复合绝缘子伞裙撕裂 4cm 应该如何处置？",
            "JN-110-052 这个杆塔有什么历史问题？帮我查一下档案和巡检记录。",
            "导线断股截面积达到多少属于 I 级缺陷？处置时效是多久？",
            "对比一下绝缘子自爆和伞裙撕裂的处置流程有什么不同？",
        ]
        selected = st.selectbox("示例问题", ["自定义输入"] + sample_qs)
        if selected == "自定义输入":
            question = st.text_area("输入问题", height=80)
        else:
            question = st.text_area("输入问题", value=selected, height=80)
        uploaded = st.file_uploader("上传巡检图像（可选）", type=["jpg", "jpeg", "png", "webp"], key="agent_qa_img")

    with col2:
        st.metric("可用工具", "4 个")
        st.caption("search_regulations\nsearch_cases\nlookup_asset\nlookup_asset_history")
        if uploaded:
            st.image(uploaded, caption="已上传图像", use_container_width=True)

    img_path = None
    if uploaded:
        img_path = str(_save_upload(uploaded))

    if st.button("Agent 推理", type="primary", disabled=not question.strip()):
        with st.spinner("Agent 推理中..."):
            t0 = time.time()
            try:
                result = run_agent(question.strip(), image_path=img_path)
            except Exception as e:
                st.error(f"Agent 执行失败: {e}")
                return
            cost = time.time() - t0

        st.success(f"完成，共 {result.total_turns} 轮推理，耗时 {cost:.1f}s")

        st.subheader("Agent 执行过程")
        _render_agent_steps(result.steps)

        st.subheader("最终回答")
        st.markdown(result.answer)


def tab_agent_diagnose():
    st.header("缺陷诊断（Agentic RAG）")
    st.caption("上传巡检图像，Agent 自动看图 + 检索规程案例 + 生成诊断。")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("上传巡检图像", type=["jpg", "jpeg", "png", "webp"])
    with col2:
        assets = _assets()
        asset_id = st.selectbox(
            "选择资产编号（可选）",
            options=["（不指定）"] + list(assets.keys()),
            index=0,
        )

    diagnosis_prompt = st.text_input(
        "补充说明（可选）",
        placeholder="例如：这是 B 相绝缘子串的特写",
    )

    if st.button("Agent 诊断", type="primary", disabled=uploaded is None):
        img_path = _save_upload(uploaded)

        prompt_parts = ["请分析这张巡检图像中的缺陷，给出诊断结论和处置建议。"]
        if asset_id != "（不指定）":
            prompt_parts.append(f"资产编号: {asset_id}，请同时查询该资产的档案和历史记录。")
        if diagnosis_prompt.strip():
            prompt_parts.append(f"补充信息: {diagnosis_prompt}")

        with st.spinner("Agent 看图 + 推理中..."):
            t0 = time.time()
            try:
                result = run_agent("\n".join(prompt_parts), image_path=str(img_path))
            except Exception as e:
                st.error(f"Agent 诊断失败: {e}")
                return
            cost = time.time() - t0

        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.image(str(img_path), caption="巡检图像", use_container_width=True)
            st.subheader("Agent 执行过程")
            _render_agent_steps(result.steps)
        with col_r:
            st.success(f"完成，{result.total_turns} 轮推理，耗时 {cost:.1f}s")
            st.subheader("诊断结论")
            st.markdown(result.answer)


def tab_langgraph_qa():
    st.header("LangGraph Agent（路由 + 质检反思）")
    st.caption("用 LangGraph StateGraph 编排的纠错式 RAG：router 入口路由 → ReAct 检索 → "
               "grade 质检 →（不足则）reflect 反思重试。比手写 ReAct 循环多了路由与自我纠错。")

    st.info("本 Tab 的回答由 **LangGraph `StateGraph`** 编排（`src/agent/graph.py`），相比"
            "「智能问答（Agent）」的手写 ReAct 循环，多了 **router / grade / reflect** 三个节点；"
            "质检不足会自动重检索（最多 2 次），故耗时通常更长。")

    _GRAPH_DOT = """digraph G {
        rankdir=LR; bgcolor="transparent"; node [fontname="sans-serif"];
        start [shape=circle,label="",width=0.25,style=filled,fillcolor="#bfb6fc"];
        router [shape=box,style="rounded,filled",fillcolor="#e8f0ff",label="router\\n意图路由"];
        agent [shape=box,style="rounded,filled",fillcolor="#f2f0ff",label="agent\\n调 MiMo + 工具"];
        tools [shape=box,style="rounded,filled",fillcolor="#f2f0ff",label="tools\\n执行检索"];
        grade [shape=box,style="rounded,filled",fillcolor="#fff0e8",label="grade\\nLLM 质检"];
        reflect [shape=box,style="rounded,filled",fillcolor="#ffe8f0",label="reflect\\n反思重试"];
        end [shape=doublecircle,label="END",style=filled,fillcolor="#bfb6fc"];
        start -> router;
        router -> agent;
        agent -> tools [label="含 tool_use",style=dashed];
        tools -> agent;
        agent -> grade [label="无 tool_use",style=dashed];
        grade -> end [label="充分 / 达上限",style=dashed];
        grade -> reflect [label="不足",style=dashed];
        reflect -> agent;
    }"""
    with st.expander(":spider_web: LangGraph 编排图（router + 条件边 + 反思循环）", expanded=True):
        st.graphviz_chart(_GRAPH_DOT, use_container_width=True)
        try:
            from src.agent.graph import _graph
            st.caption("以下为 LangGraph 由编译后的图导出的 Mermaid 源（证明上图来自 StateGraph 本身）：")
            st.code(_graph().get_graph().draw_mermaid(), language="text")
        except Exception as e:  # 版本差异时降级，不影响问答
            st.caption(f"Mermaid 源导出不可用：{e}")

    sample_qs = [
        "复合绝缘子伞裙撕裂 4cm 应该如何处置？",
        "JN-110-052 这个杆塔有什么历史问题？帮我查一下档案和巡检记录。",
        "导线断股截面积达到多少属于 I 级缺陷？处置时效是多久？",
        "对比一下绝缘子自爆和伞裙撕裂的处置流程有什么不同？",
    ]
    selected = st.selectbox("示例问题", ["自定义输入"] + sample_qs, key="lg_qa_select")
    if selected == "自定义输入":
        question = st.text_area("输入问题", height=80, key="lg_qa_input")
    else:
        question = st.text_area("输入问题", value=selected, height=80, key="lg_qa_input")

    if st.button("LangGraph 推理", type="primary", disabled=not question.strip(), key="lg_qa_btn"):
        with st.spinner("LangGraph 编排执行中..."):
            t0 = time.time()
            try:
                result = run_graph_agent(question.strip())
            except Exception as e:
                st.error(f"LangGraph 执行失败: {e}")
                return
            cost = time.time() - t0

        st.success(f"完成，共 {result.total_turns} 轮推理，耗时 {cost:.1f}s")
        st.subheader("Agent 执行过程")
        _render_agent_steps(result.steps)
        st.subheader("最终回答")
        st.markdown(result.answer)


def tab_report():
    st.header("巡检报告生成")
    st.caption("基于本会话累计的诊断结果，自动生成结构化报告草稿。")

    diagnoses = st.session_state.get("diagnoses", [])
    if not diagnoses:
        st.warning("当前会话还没有诊断记录，请先到「缺陷诊断」tab 上传图像并诊断。")
        return

    st.write(f"已累计诊断 {len(diagnoses)} 条")
    col1, col2, col3 = st.columns(3)
    with col1:
        ins_id = st.text_input("巡检任务编号", value=f"INS-{date.today().strftime('%Y%m%d')}-DEMO")
    with col2:
        ins_date = st.text_input("巡检日期", value=date.today().isoformat())
    with col3:
        method = st.selectbox("巡检方式", ["无人机精细化巡视", "登塔检查", "红外测温", "无人机绕飞"])

    if st.button("生成报告草稿", type="primary"):
        with st.spinner("聚合诊断 -> LLM 报告生成 ..."):
            try:
                report_md = generate_report(ins_id, ins_date, method, diagnoses)
            except Exception as e:
                st.error(f"报告生成失败: {e}")
                return
        st.markdown(report_md)
        st.download_button(
            "下载 Markdown",
            data=report_md.encode("utf-8"),
            file_name=f"{ins_id}.md",
            mime="text/markdown",
        )


def tab_basic_qa():
    st.header("知识问答（基础 RAG）")
    st.caption("传统 RAG 流程：固定检索 -> 生成，可与 Agent 模式对比效果。")

    question = st.text_area("输入问题", height=80, key="basic_qa_input")
    if st.button("提问", type="primary", disabled=not question.strip(), key="basic_qa_btn"):
        with st.spinner("检索 + 生成中..."):
            t0 = time.time()
            try:
                resp = answer_question(question.strip())
            except Exception as e:
                st.error(f"问答失败: {e}")
                return
            cost = time.time() - t0
        st.success(f"完成，耗时 {cost:.1f}s")
        st.subheader("回答")
        st.markdown(resp["answer"])
        st.subheader("引用上下文")
        for i, c in enumerate(resp["contexts"], 1):
            meta = c.get("metadata", {})
            score = c.get("fused_score") or c.get("score", 0)
            with st.expander(f"#{i} {meta.get('source') or meta.get('case_id', c.get('id'))} | 相似度 {score:.3f}"):
                st.write(c.get("document", "")[:800])


def tab_system():
    st.header("系统信息")
    cfg = load_config()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("模型配置")
        st.json({
            "LLM": cfg["provider"]["llm_model"],
            "Embedding": cfg["provider"]["embedding_model"] + " (本地)",
            "Base URL": cfg["_env"]["llm_base_url"],
            "协议": "Anthropic",
        })
        st.subheader("Agent 工具")
        st.json([
            "search_regulations — 检索行业规程",
            "search_cases — 检索历史案例",
            "lookup_asset — 查询资产档案",
            "lookup_asset_history — 查询巡检历史",
        ])
    with col2:
        st.subheader("数据规模")
        chunks = load_regulation_chunks()
        cases = load_defect_cases()
        history = load_inspection_history()
        assets = load_assets()
        st.metric("规程 chunk 数", len(chunks))
        st.metric("历史缺陷案例", len(cases))
        st.metric("资产档案", len(assets))
        st.metric("巡检历史", len(history))


def main():
    st.title("无人机巡检 Agentic RAG 系统")
    st.caption("v2.0 | Agent 自主推理 + 多工具协同 + 引用溯源")

    tabs = st.tabs([
        "智能问答（Agent）", "缺陷诊断（Agent）", "LangGraph Agent",
        "巡检报告", "基础RAG对比", "系统信息",
    ])
    with tabs[0]:
        tab_agent_qa()
    with tabs[1]:
        tab_agent_diagnose()
    with tabs[2]:
        tab_langgraph_qa()
    with tabs[3]:
        tab_report()
    with tabs[4]:
        tab_basic_qa()
    with tabs[5]:
        tab_system()


if __name__ == "__main__":
    main()
