"""Streamlit Demo 前端

八个 Tab：
1. 智能问答（Agent）    - Agentic RAG，LLM 自主调用工具
2. 缺陷诊断（Agent）    - 上传图像，Agent 自主看图 + 检索 + 诊断（需配置多模态模型）
3. LangGraph Agent      - 路由分流 + 质检反思的纠错式编排
4. 低空作业台           - 复飞任务规划 + 飞行前合规校验（确定性，不调模型）
5. 多智能体协作         - 规划员 → 合规闸门 → 诊断员 → 调度员
6. 巡检报告             - 多条诊断聚合生成报告草稿
7. 基础 RAG 对比        - 传统 RAG 流程对照
8. 系统信息             - 配置 / 数据规模 / 工具清单
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
from src.agent.crew import run_crew
from src.agent.graph import run_graph_agent
from src.agent.tools import TOOL_NAMES
from src.config import load_config, vlm_available
from src.generation.report_generator import DiagnosisResult, answer_question, generate_report
from src.ingestion.text_loader import (
    load_assets,
    load_defect_cases,
    load_inspection_history,
    load_regulation_chunks,
)
from src.mission.airspace import check_flight, format_report
from src.mission.planner import format_plan, plan_mission

st.set_page_config(page_title="无人机巡检 Agentic RAG", page_icon=":mag:", layout="wide")

TOOL_ICONS = {
    "search_regulations": ":book:",
    "search_cases": ":file_folder:",
    "lookup_asset": ":wrench:",
    "lookup_asset_history": ":clock3:",
    "plan_inspection_mission": ":round_pushpin:",
    "check_flight_clearance": ":vertical_traffic_light:",
}

SAMPLE_QUESTIONS = [
    "复合绝缘子伞裙撕裂 4cm 应该如何处置？",
    "JN-110-052 这个杆塔有什么历史问题？帮我查一下档案和巡检记录。",
    "导线断股截面积达到多少属于 I 级缺陷？处置时效是多久？",
    "济南西郊 110kV 输电线路明天该复飞哪些杆塔？怎么排架次？",
    "QD-110-099 和 QD-110-100 今天能不能飞？有没有空域限制？",
]


@st.cache_data(show_spinner=False)
def _assets() -> dict:
    return load_assets()


@st.cache_data(show_spinner=False)
def _line_names() -> list[str]:
    return sorted({a["line_name"] for a in _assets().values()})


def _save_upload(file) -> Path:
    suffix = Path(file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir())
    tmp.write(file.getvalue())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _image_uploader(key: str, label: str = "上传巡检图像（可选）"):
    """多模态未配置时不提供上传入口，避免用户点了才报错。"""
    if not vlm_available():
        st.caption(":no_entry_sign: 未配置多模态模型（config.yaml 的 `vlm:`），图像输入已禁用；纯文本功能不受影响。")
        return None
    return st.file_uploader(label, type=["jpg", "jpeg", "png", "webp"], key=key)


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


def _run_and_render(runner, question: str, img_path: str | None, label: str):
    with st.spinner(f"{label}执行中..."):
        t0 = time.time()
        try:
            result = runner(question.strip(), image_path=img_path)
        except Exception as e:
            st.error(f"{label}失败: {e}")
            return
        cost = time.time() - t0
    st.success(f"完成，共 {result.total_turns} 轮推理，耗时 {cost:.1f}s")
    st.subheader("Agent 执行过程")
    _render_agent_steps(result.steps)
    st.subheader("最终回答")
    st.markdown(result.answer)
    return result


def tab_agent_qa():
    st.header("智能问答（Agentic RAG）")
    st.caption("Agent 自主决定调用哪些工具，支持多轮推理。工具既有语义检索，也有确定性的规划与合规计算。")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("示例问题", ["自定义输入"] + SAMPLE_QUESTIONS)
        value = "" if selected == "自定义输入" else selected
        question = st.text_area("输入问题", value=value, height=80)
        uploaded = _image_uploader("agent_qa_img")
    with col2:
        st.metric("可用工具", f"{len(TOOL_NAMES)} 个")
        st.caption("\n".join(TOOL_NAMES))
        if uploaded:
            st.image(uploaded, caption="已上传图像", use_container_width=True)

    img_path = str(_save_upload(uploaded)) if uploaded else None
    if st.button("Agent 推理", type="primary", disabled=not question.strip()):
        _run_and_render(run_agent, question, img_path, "Agent 推理")


def tab_agent_diagnose():
    st.header("缺陷诊断（Agentic RAG）")
    st.caption("上传巡检图像，Agent 自动看图 + 检索规程案例 + 生成诊断。")

    if not vlm_available():
        st.warning(
            "本 Tab 需要多模态模型。请在 `config.yaml` 中把 `vlm.enabled` 设为 true、"
            "填入 `vlm.model` / `vlm.base_url`，并在 `.env` 里配置对应的 API Key。"
        )
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("上传巡检图像", type=["jpg", "jpeg", "png", "webp"])
    with col2:
        asset_id = st.selectbox("选择资产编号（可选）", ["（不指定）"] + list(_assets()))

    extra = st.text_input("补充说明（可选）", placeholder="例如：这是 B 相绝缘子串的特写")

    if st.button("Agent 诊断", type="primary", disabled=uploaded is None):
        img_path = _save_upload(uploaded)
        parts = ["请分析这张巡检图像中的缺陷，给出诊断结论和处置建议。"]
        if asset_id != "（不指定）":
            parts.append(f"资产编号: {asset_id}，请同时查询该资产的档案和历史记录。")
        if extra.strip():
            parts.append(f"补充信息: {extra}")

        st.image(str(img_path), caption="巡检图像", width=420)
        result = _run_and_render(run_agent, "\n".join(parts), str(img_path), "Agent 诊断")
        if result:
            # 存进会话，供「巡检报告」Tab 聚合成报告草稿
            st.session_state.setdefault("diagnoses", []).append(DiagnosisResult(
                image_path=str(img_path),
                detection={},
                asset_card=_assets().get(asset_id) if asset_id != "（不指定）" else None,
                asset_history=[],
                regulation_hits=[],
                case_hits=[],
                diagnosis_md=result.answer,
            ))


GRAPH_DOT = """digraph G {
    rankdir=LR; bgcolor="transparent"; node [fontname="sans-serif"];
    start [shape=circle,label="",width=0.25,style=filled,fillcolor="#bfb6fc"];
    router [shape=diamond,style=filled,fillcolor="#e8f0ff",label="router\\n意图分流"];
    direct [shape=box,style="rounded,filled",fillcolor="#e8ffe8",label="direct_lookup\\n确定性快路径\\n(档案/历史/合规/排期)"];
    agent [shape=box,style="rounded,filled",fillcolor="#f2f0ff",label="agent\\n调 LLM + 工具"];
    tools [shape=box,style="rounded,filled",fillcolor="#f2f0ff",label="tools\\n执行工具"];
    grade [shape=box,style="rounded,filled",fillcolor="#fff0e8",label="grade\\nLLM 质检"];
    reflect [shape=box,style="rounded,filled",fillcolor="#ffe8f0",label="reflect\\n反思重试"];
    end [shape=doublecircle,label="END",style=filled,fillcolor="#bfb6fc"];
    start -> router;
    router -> direct [label="工具确定的场景",style=dashed];
    router -> agent [label="需自主编排",style=dashed];
    direct -> agent;
    agent -> tools [label="含 tool_call",style=dashed];
    tools -> agent;
    agent -> grade [label="无 tool_call",style=dashed];
    grade -> end [label="通过 / 反思用尽",style=dashed];
    grade -> reflect [label="不足",style=dashed];
    reflect -> agent;
}"""


def tab_langgraph_qa():
    st.header("LangGraph Agent（路由分流 + 质检反思）")
    st.caption("用 LangGraph StateGraph 编排的纠错式 RAG：router 按意图分流 → grade 质检 →（不足则）reflect 反思重试。")

    st.info(
        "router 是**真条件分支**：资产历史查询、飞行前合规校验、指名线路的排期这三类"
        "「所需工具已确定」的问题走确定性快路径（跳过 LLM 选工具的来回），其余走 ReAct；"
        "答案再经 grade 质检，不足则 reflect 重检索。"
    )

    with st.expander(":spider_web: LangGraph 编排图", expanded=True):
        st.graphviz_chart(GRAPH_DOT, use_container_width=True)
        try:
            from src.agent.graph import graph

            st.caption("以下为 LangGraph 从编译后的图导出的 Mermaid 源（证明上图来自 StateGraph 本身）：")
            st.code(graph().get_graph().draw_mermaid(), language="text")
        except Exception as e:  # 版本差异时降级，不影响问答
            st.caption(f"Mermaid 源导出不可用：{e}")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("示例问题", ["自定义输入"] + SAMPLE_QUESTIONS, key="lg_qa_select")
        value = "" if selected == "自定义输入" else selected
        question = st.text_area("输入问题", value=value, height=80, key="lg_qa_input")
        uploaded = _image_uploader("lg_qa_img")
    with col2:
        st.metric("图编排节点", "6 个")
        st.caption("router / direct_lookup\nagent / tools\ngrade / reflect")
        if uploaded:
            st.image(uploaded, caption="已上传图像", use_container_width=True)

    img_path = str(_save_upload(uploaded)) if uploaded else None
    if st.button("LangGraph 推理", type="primary", disabled=not question.strip(), key="lg_qa_btn"):
        _run_and_render(run_graph_agent, question, img_path, "LangGraph 编排")


def _weather_inputs(key_prefix: str) -> dict:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        condition = st.selectbox("天气现象", ["晴", "多云", "阴", "小雨", "雷暴", "雾"], key=f"{key_prefix}_cond")
    with c2:
        wind = st.number_input("平均风速 m/s", 0.0, 30.0, 5.0, 0.5, key=f"{key_prefix}_wind")
    with c3:
        vis = st.number_input("能见度 km", 0.0, 30.0, 10.0, 0.5, key=f"{key_prefix}_vis")
    with c4:
        temp = st.number_input("气温 ℃", -30.0, 45.0, 15.0, 1.0, key=f"{key_prefix}_temp")
    return {"condition": condition, "wind_mps": wind, "visibility_km": vis, "temperature_c": temp}


def tab_mission():
    st.header("低空作业台（确定性计算，不调模型）")
    st.caption(
        "复飞任务规划与飞行前合规校验都是可验证计算：时效判定、航程与续航测算、空域限高比对。"
        "把它们从模型手里拿走，是为了让这部分零幻觉、可单测——模型只负责解释结果。"
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        line = st.selectbox("线路", ["全部线路"] + _line_names())
    with c2:
        ref = st.date_input("参考日期", value=date(2025, 10, 1))
    with c3:
        agl = st.number_input("计划作业真高 m", 10.0, 300.0, 120.0, 10.0)

    weather = _weather_inputs("mission")

    if st.button("生成任务计划并校验", type="primary"):
        plan = plan_mission(ref.isoformat(), None if line == "全部线路" else line)
        left, right = st.columns(2)
        with left:
            st.subheader("① 复飞任务规划")
            m1, m2, m3 = st.columns(3)
            m1.metric("待飞杆塔", len(plan.tasks))
            m2.metric("架次", len(plan.sorties))
            m3.metric("顺延", len(plan.deferred))
            st.markdown(format_plan(plan))
        with right:
            st.subheader("② 飞行前合规校验")
            ids = [aid for s in plan.sorties for aid in s.asset_ids]
            if not ids:
                st.info("无待飞塔位，跳过合规校验。")
            else:
                report = check_flight(ids, agl_m=agl, weather=weather)
                badge = {"allowed": ":white_check_mark: 放行",
                         "restricted": ":warning: 有条件放行",
                         "forbidden": ":no_entry: 禁止起飞"}[report.verdict]
                st.markdown(f"### {badge}")
                st.markdown(format_report(report))


def tab_crew():
    st.header("多智能体协作（规划员 → 合规闸门 → 诊断员 → 调度员）")
    st.caption(
        "每个角色只看得见自己该用的工具子集，上一棒的产物写进共享状态传给下一棒。"
        "中间的合规闸门是确定性判定：全员禁飞时直接跳过诊断，让调度员出改期方案。"
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        line = st.selectbox("线路", _line_names(), key="crew_line")
    with c2:
        ref = st.date_input("作业日期", value=date(2025, 10, 1), key="crew_date")
    with c3:
        agl = st.number_input("计划作业真高 m", 10.0, 300.0, 120.0, 10.0, key="crew_agl")
    weather = _weather_inputs("crew")

    if st.button("启动班组协作", type="primary", key="crew_btn"):
        with st.spinner("规划员 → 合规闸门 → 诊断员 → 调度员..."):
            t0 = time.time()
            try:
                result = run_crew(line, ref.isoformat(), weather=weather, agl_m=agl)
            except Exception as e:
                st.error(f"多智能体执行失败: {e}")
                return
            cost = time.time() - t0

        st.success(f"完成，耗时 {cost:.1f}s｜待飞 {len(result.asset_ids)} 基｜合规闸门：{result.clearance_verdict}")
        st.subheader("协作过程")
        _render_agent_steps(result.steps)

        for title, body in [
            (":round_pushpin: 规划员产出", result.plan_md),
            (":vertical_traffic_light: 合规闸门（确定性）", result.clearance_md),
            (":microscope: 诊断员产出", result.diagnosis_md or "（本次跳过逐基诊断）"),
            (":clipboard: 调度员派工单", result.dispatch_md),
        ]:
            with st.expander(title, expanded=title.endswith("派工单")):
                st.markdown(body)


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
        st.download_button("下载 Markdown", data=report_md.encode("utf-8"),
                           file_name=f"{ins_id}.md", mime="text/markdown")


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
            title = meta.get("source") or meta.get("case_id") or c.get("id")
            with st.expander(f"#{i} {title} | 相似度 {score:.3f}"):
                st.write(c.get("document", "")[:800])


def tab_system():
    st.header("系统信息")
    cfg = load_config()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("模型配置")
        st.json({
            "文本模型": f"{cfg['llm']['provider']} / {cfg['llm']['model']}",
            "多模态模型": (f"{cfg['vlm']['provider']} / {cfg['vlm']['model']}"
                           if vlm_available() else "未配置（图像功能已禁用）"),
            "Embedding": cfg["embedding"]["model"] + "（本地）",
            "ReAct 最大轮次": cfg["agent"]["max_turns"],
            "最大反思次数": cfg["agent"]["max_reflections"],
        })
        st.subheader("Agent 工具")
        st.json(TOOL_NAMES)
    with col2:
        st.subheader("数据规模")
        st.metric("规程 chunk 数", len(load_regulation_chunks()))
        st.metric("历史缺陷案例", len(load_defect_cases()))
        st.metric("资产档案（杆塔）", len(load_assets()))
        st.metric("巡检历史", len(load_inspection_history()))
        st.subheader("低空作业参数")
        st.json(cfg["mission"])


def main():
    st.title("无人机巡检 Agentic RAG 系统")
    st.caption("Agent 自主推理 + 多工具协同 + 确定性作业计算 + 引用溯源")

    names = ["智能问答（Agent）", "缺陷诊断（Agent）", "LangGraph Agent", "低空作业台",
             "多智能体协作", "巡检报告", "基础RAG对比", "系统信息"]
    fns = [tab_agent_qa, tab_agent_diagnose, tab_langgraph_qa, tab_mission,
           tab_crew, tab_report, tab_basic_qa, tab_system]
    for tab, fn in zip(st.tabs(names), fns, strict=True):
        with tab:
            fn()


if __name__ == "__main__":
    main()
