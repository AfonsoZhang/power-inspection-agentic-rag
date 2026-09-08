"""Agentic RAG 工具定义（provider 中立格式）

工具分两类，这个划分本身就是设计取舍：

- **检索类**（search_regulations / search_cases / lookup_asset / lookup_asset_history）
  面向"知识在哪"，结果需要模型综合表述。
- **确定性作业类**（plan_inspection_mission / check_flight_clearance）
  面向"算出来的答案"——时效判定、航程与续航测算、空域限高比对，全部是可验证计算。
  把它们做成工具而不是写进提示词让模型推，是为了让这部分**零幻觉且可单测**；
  模型只决定何时调用、以及如何向人解释结果。
- **策略评估类**（simulate_flight_policy / optimize_flight_policy）
  面向"这么排靠不靠谱"——把作业策略图丢进内置仿真做蒙特卡洛推演，或在仿真里搜更优的图。
  同样零模型参与，见 src/policy/。

工具 schema 用中立格式 {name, description, parameters}，由 providers.py 翻译成
Anthropic 的 input_schema 或 OpenAI 的 function.parameters。
"""
from __future__ import annotations

import json
from datetime import date

TOOL_DEFINITIONS = [
    {
        "name": "search_regulations",
        "description": "检索电力巡检行业规程（绝缘子、杆塔、导线金具等），返回相关条款片段。用于回答处置标准、缺陷分级、时效要求等问题。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词，例如'绝缘子自爆处置'、'导线断股分级标准'",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_cases",
        "description": "检索历史缺陷案例库，返回相似的处置案例。可按缺陷类型或资产类型过滤。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "案例检索关键词，例如'复合绝缘子伞裙撕裂'",
                },
                "defect_type": {
                    "type": "string",
                    "description": "可选，按缺陷类型过滤，例如'螺栓松动'、'绝缘子单片自爆'",
                },
                "asset_type": {
                    "type": "string",
                    "description": "可选，按资产类型过滤，例如'110kV角钢塔'、'220kV角钢塔'",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "lookup_asset",
        "description": "根据资产编号查询设备档案卡，包含线路名称、型号、投运年份、经纬度、责任人等信息。",
        "parameters": {
            "type": "object",
            "properties": {
                "asset_id": {
                    "type": "string",
                    "description": "资产编号，格式如 JN-110-052、QD-110-103",
                },
            },
            "required": ["asset_id"],
        },
    },
    {
        "name": "lookup_asset_history",
        "description": "查询某资产的历史巡检记录，了解该设备过去的巡检情况和发现的问题。",
        "parameters": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string", "description": "资产编号"},
                "limit": {"type": "integer", "description": "返回最近几条记录，默认5"},
            },
            "required": ["asset_id"],
        },
    },
    {
        "name": "plan_inspection_mission",
        "description": (
            "生成无人机复飞巡检任务计划：按缺陷等级对应的规程时效、以及例行巡检间隔筛出待飞杆塔，"
            "同优先级组内做航线优化，再按电池续航切分架次。"
            "用户问「哪些塔该复飞 / 明天怎么排架次 / 这条线路的巡检计划」时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "line_name": {
                    "type": "string",
                    "description": "线路名称，例如'济南西郊 110kV 输电线路'。留空表示全部线路。",
                },
                "reference_date": {
                    "type": "string",
                    "description": "计划参考日期 YYYY-MM-DD，缺省为今天",
                },
            },
            "required": [],
        },
    },
    {
        "name": "check_flight_clearance",
        "description": (
            "无人机起飞前合规校验：比对空域限制区（净空区/禁飞区/限高区）、气象限值与带电体安全距离，"
            "给出放行/有条件放行/禁止起飞的结论与处置建议。"
            "用户问「这些塔能不能飞 / 今天天气能飞吗 / 有没有空域限制」时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "asset_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "待校验的杆塔编号列表",
                },
                "agl_m": {
                    "type": "number",
                    "description": "计划作业真高（米），缺省 120",
                },
                "weather": {
                    "type": "object",
                    "description": (
                        "起飞前气象实况，字段：condition(天气现象文字)、wind_mps、gust_mps、"
                        "visibility_km、temperature_c。不提供则气象一项判为 unknown。"
                    ),
                },
            },
            "required": ["asset_ids"],
        },
    },
    {
        "name": "simulate_flight_policy",
        "description": (
            "对当前作业策略图做蒙特卡洛仿真，评估这份排班在风况、电池健康、悬停时长分散、"
            "临时发现缺陷四种扰动下的完成率、架次中断率与备降风险。"
            "用户问「这么排靠谱吗 / 风大了还飞得完吗 / 会不会飞一半电量不够」时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "line_name": {"type": "string", "description": "线路名称，留空表示全部线路"},
                "reference_date": {"type": "string", "description": "计划参考日期 YYYY-MM-DD，缺省为今天"},
                "weather": {
                    "type": "object",
                    "description": "气象实况，字段同 check_flight_clearance；缺省按 5 m/s 晴天推演",
                },
                "trials": {"type": "integer", "description": "蒙特卡洛次数，缺省 30"},
            },
            "required": [],
        },
    },
    {
        "name": "optimize_flight_policy",
        "description": (
            "在仿真中搜索更优的作业策略图：遍历航线算法、电池安全余量、单架次塔位上限、"
            "返航阈值等组合，按「安全 → 完成率 → 不中断率 → 吞吐」挑出最优图，并给出与基线的对照。"
            "用户问「怎么排能多飞几基 / 参数该怎么调 / 有没有更稳的排法」时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "line_name": {"type": "string", "description": "线路名称，留空表示全部线路"},
                "reference_date": {"type": "string", "description": "计划参考日期 YYYY-MM-DD，缺省为今天"},
                "weather": {"type": "object", "description": "气象实况；缺省按 5 m/s 晴天推演"},
                "trials": {"type": "integer", "description": "单张候选图的蒙特卡洛次数，缺省 24"},
            },
            "required": [],
        },
    },
]

TOOL_NAMES = [t["name"] for t in TOOL_DEFINITIONS]

# 仿真的默认气象：晴、5 m/s。写死默认值是为了让两次调用的结果可比——
# 用户没给天气时不能随机取一个，否则"改进了没有"这个问题没法回答。
DEFAULT_SIM_WEATHER = {
    "condition": "晴", "wind_mps": 5.0, "gust_mps": 7.0,
    "visibility_km": 10.0, "temperature_c": 15.0,
}


def execute_tool(name: str, args: dict) -> str:
    """执行工具调用，返回结果文本。任何异常都转成可读文本回灌给模型，避免整轮崩掉。"""
    try:
        return _dispatch(name, args)
    except Exception as e:  # noqa: BLE001 - 工具错误要让模型看见并自行纠正
        return f"工具 {name} 执行失败：{type(e).__name__}: {e}"


def _dispatch(name: str, args: dict) -> str:
    # 检索/规划模块都带重依赖，按需 import，保证只做 schema 校验的测试无需装模型栈
    if name == "search_regulations":
        from ..retrieval.retriever import retrieve_regulations

        return _format_hits(retrieve_regulations(args["query"]), kind="规程条款")

    if name == "search_cases":
        from ..retrieval.retriever import retrieve_cases

        hits = retrieve_cases(
            args["query"],
            defect_type=args.get("defect_type"),
            asset_type=args.get("asset_type"),
        )
        return _format_hits(hits, kind="历史案例")

    if name == "lookup_asset":
        from ..retrieval.retriever import retrieve_asset_card

        card = retrieve_asset_card(args["asset_id"])
        if not card:
            return f"未找到资产 {args['asset_id']} 的档案信息。"
        return json.dumps(card, ensure_ascii=False, indent=2)

    if name == "lookup_asset_history":
        from ..retrieval.retriever import retrieve_asset_history

        history = retrieve_asset_history(args["asset_id"], limit=int(args.get("limit") or 5))
        if not history:
            return f"资产 {args['asset_id']} 无历史巡检记录。"
        return json.dumps(history, ensure_ascii=False, indent=2)

    if name == "plan_inspection_mission":
        from ..mission.planner import format_plan, plan_mission

        ref = args.get("reference_date") or date.today().isoformat()
        return format_plan(plan_mission(ref, args.get("line_name") or None))

    if name == "check_flight_clearance":
        from ..mission.airspace import check_flight, format_report

        asset_ids = args.get("asset_ids") or []
        if not asset_ids:
            return "未提供任何杆塔编号，无法校验。请先确定待飞塔位。"
        report = check_flight(
            list(asset_ids),
            agl_m=args.get("agl_m"),
            weather=args.get("weather"),
        )
        return format_report(report)

    if name == "simulate_flight_policy":
        from ..policy.policy_graph import active_policy
        from ..policy.simulator import format_sim, simulate

        return format_sim(simulate(
            active_policy(),
            args.get("reference_date") or date.today().isoformat(),
            args.get("line_name") or None,
            args.get("weather") or DEFAULT_SIM_WEATHER,
            trials=int(args.get("trials") or 30),
        ))

    if name == "optimize_flight_policy":
        from ..policy.selflearn import format_learning, search

        return format_learning(search(
            args.get("reference_date") or date.today().isoformat(),
            args.get("line_name") or None,
            args.get("weather") or DEFAULT_SIM_WEATHER,
            trials=int(args.get("trials") or 24),
        ))

    return f"未知工具: {name}"


def _format_hits(hits: list[dict], *, kind: str) -> str:
    if not hits:
        return f"未找到相关{kind}。"
    parts = []
    for h in hits:
        meta = h.get("metadata") or {}
        head_bits = [
            str(meta.get("source") or meta.get("case_id") or h.get("id") or ""),
            str(meta.get("section_path") or meta.get("defect_type") or ""),
        ]
        score = h.get("score")
        if isinstance(score, float):
            head_bits.append(f"相似度={score:.3f}")
        parts.append(" | ".join(b for b in head_bits if b) + "\n" + h.get("document", "")[:500])
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(parts, 1))
