"""复飞任务规划：从「缺陷 + 规程时效 + 巡检历史」推导出无人机巡检架次计划

这一层是**确定性**的（不调 LLM），Agent 只负责决定「什么时候、对哪条线路」调用它，
结果的正确性由单元测试保证——这也是把它做成工具而不是让模型自由发挥的原因：
时效判定和航程/续航测算属于可验证计算，交给模型只会引入不可控误差。

流程：
    1. 需求识别  缺陷复查（按等级取时效）+ 例行巡检（按间隔）
    2. 优先级排序 I > II > III > 例行；同级按到期日升序
    3. 航线规划  最近邻 + 2-opt 求访问顺序
    4. 架次切分  按电池有效作业时长切分，超出单日架次上限的推入待排期
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta

from ..config import load_config
from ..ingestion.text_loader import load_assets, load_defect_cases, load_inspection_history
from .geo import haversine_km, optimize_route

SEVERITY_RANK = {"I": 0, "II": 1, "III": 2}


@dataclass
class MissionTask:
    """一基待飞杆塔。"""

    asset_id: str
    line_name: str
    lat: float
    lon: float
    task_type: str           # "defect_recheck" | "routine"
    priority: str            # "I" | "II" | "III" | "routine"
    reason: str
    due_date: str            # ISO 日期
    overdue_days: int        # >0 表示已超期
    source_case_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Sortie:
    """一个架次（一块电池能覆盖的连续作业段）。"""

    index: int
    asset_ids: list[str] = field(default_factory=list)
    transit_km: float = 0.0
    hover_minutes: float = 0.0
    flight_minutes: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MissionPlan:
    reference_date: str
    line_name: str
    tasks: list[MissionTask]
    sorties: list[Sortie]
    deferred: list[MissionTask]      # 超出单日架次能力，需顺延
    total_km: float
    total_minutes: float

    def to_dict(self) -> dict:
        return {
            "reference_date": self.reference_date,
            "line_name": self.line_name,
            "tasks": [t.to_dict() for t in self.tasks],
            "sorties": [s.to_dict() for s in self.sorties],
            "deferred": [t.to_dict() for t in self.deferred],
            "total_km": round(self.total_km, 3),
            "total_minutes": round(self.total_minutes, 1),
        }


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _latest_inspection(history: list[dict]) -> dict[str, date]:
    latest: dict[str, date] = {}
    for h in history:
        d = _parse_date(h["date"])
        if h["asset_id"] not in latest or d > latest[h["asset_id"]]:
            latest[h["asset_id"]] = d
    return latest


def _open_defects(cases: list[dict], history: list[dict]) -> dict[str, list[dict]]:
    """按资产归集**未闭环**缺陷。

    闭环判据：该缺陷发现日之后存在更晚的巡检记录，且那次记录的 findings 里不再出现该 case_id
    —— 复查过且没再复现，视为已消缺。数据里没有独立的消缺台账，这是能从现有字段推出的最强证据。
    """
    later_findings: dict[str, list[tuple[date, set[str]]]] = {}
    for h in history:
        later_findings.setdefault(h["asset_id"], []).append(
            (_parse_date(h["date"]), set(h.get("findings") or []))
        )

    by_asset: dict[str, list[dict]] = {}
    for c in cases:
        aid = c["asset_id"]
        found = _parse_date(c["discovery_date"])
        closed = any(
            d > found and c["case_id"] not in findings
            for d, findings in later_findings.get(aid, [])
        )
        if not closed:
            by_asset.setdefault(aid, []).append(c)
    return by_asset


def collect_tasks(reference_date: str, line_name: str | None = None) -> list[MissionTask]:
    """识别到参考日期为止需要出动无人机的杆塔。"""
    cfg = load_config()
    mission_cfg = cfg["mission"]
    recheck_days = mission_cfg["recheck_days_by_severity"]
    routine_interval = mission_cfg["routine_interval_days"]

    ref = _parse_date(reference_date)
    assets = load_assets()
    cases = load_defect_cases()
    history = load_inspection_history()
    latest = _latest_inspection(history)
    defects = _open_defects(cases, history)

    tasks: list[MissionTask] = []
    for aid, asset in assets.items():
        if line_name and asset["line_name"] != line_name:
            continue
        loc = asset["location"]

        asset_defects = defects.get(aid, [])
        if asset_defects:
            # 取最紧急的一条缺陷（等级优先、其次发现日期新）驱动复查
            worst = min(
                asset_defects,
                key=lambda c: (SEVERITY_RANK.get(c["severity"], 9), -_parse_date(c["discovery_date"]).toordinal()),
            )
            sev = worst["severity"]
            due = _parse_date(worst["discovery_date"]) + timedelta(days=int(recheck_days.get(sev, 30)))
            tasks.append(MissionTask(
                asset_id=aid, line_name=asset["line_name"], lat=loc["lat"], lon=loc["lon"],
                task_type="defect_recheck", priority=sev,
                reason=f"{sev} 级缺陷「{worst['defect_type']}」（{worst['case_id']}，发现于 {worst['discovery_date']}）需复查",
                due_date=due.isoformat(), overdue_days=max(0, (ref - due).days),
                source_case_id=worst["case_id"],
            ))
            continue

        last = latest.get(aid)
        due = (last + timedelta(days=routine_interval)) if last else ref
        if due <= ref:
            reason = (f"距上次巡检 {(ref - last).days} 天，超过例行间隔 {routine_interval} 天"
                      if last else "无巡检记录，需首次建档巡视")
            tasks.append(MissionTask(
                asset_id=aid, line_name=asset["line_name"], lat=loc["lat"], lon=loc["lon"],
                task_type="routine", priority="routine", reason=reason,
                due_date=due.isoformat(), overdue_days=max(0, (ref - due).days),
            ))

    tasks.sort(key=lambda t: (SEVERITY_RANK.get(t.priority, 3), t.due_date, t.asset_id))
    return tasks


def split_into_sorties(tasks: list[MissionTask]) -> tuple[list[Sortie], list[MissionTask]]:
    """按航程与电池续航把有序任务切成架次；超出单日架次上限的部分作为待顺延返回。"""
    cfg = load_config()["mission"]
    speed_m_per_min = cfg["cruise_speed_mps"] * 60.0
    hover = cfg["hover_minutes_per_tower"]
    battery = cfg["battery_minutes"]
    max_sorties = cfg["max_sorties_per_day"]

    sorties: list[Sortie] = []
    current = Sortie(index=1)
    prev: MissionTask | None = None

    for task in tasks:
        transit_km = haversine_km((prev.lat, prev.lon), (task.lat, task.lon)) if prev else 0.0
        transit_min = transit_km * 1000.0 / speed_m_per_min
        needed = current.flight_minutes + transit_min + hover

        if current.asset_ids and needed > battery:
            sorties.append(current)
            current = Sortie(index=len(sorties) + 1)
            transit_km, transit_min = 0.0, 0.0  # 新架次从起降点重新起飞
            needed = hover

        current.asset_ids.append(task.asset_id)
        current.transit_km += transit_km
        current.hover_minutes += hover
        current.flight_minutes = needed
        prev = task

    if current.asset_ids:
        sorties.append(current)

    if len(sorties) <= max_sorties:
        return sorties, []

    kept = sorties[:max_sorties]
    scheduled = {aid for s in kept for aid in s.asset_ids}
    deferred = [t for t in tasks if t.asset_id not in scheduled]
    return kept, deferred


def plan_mission(reference_date: str, line_name: str | None = None) -> MissionPlan:
    """端到端生成一条线路的复飞任务计划。"""
    tasks = collect_tasks(reference_date, line_name)
    if not tasks:
        return MissionPlan(reference_date, line_name or "全部线路", [], [], [], 0.0, 0.0)

    # 航程优化只在**同优先级组内**做：时效是硬约束（I 级不能因为顺路被排到 III 级后面），
    # 省航程是软目标，所以按优先级分组、组内各自 2-opt，再按优先级顺序拼接。
    ordered: list[MissionTask] = []
    for priority in ("I", "II", "III", "routine"):
        group = [t for t in tasks if t.priority == priority]
        if not group:
            continue
        order, _ = optimize_route([(t.lat, t.lon) for t in group], start=0)
        ordered.extend(group[i] for i in order)

    sorties, deferred = split_into_sorties(ordered)
    scheduled_ids = {aid for s in sorties for aid in s.asset_ids}
    ordered_scheduled = [t for t in ordered if t.asset_id in scheduled_ids]

    return MissionPlan(
        reference_date=reference_date,
        line_name=line_name or "全部线路",
        tasks=ordered_scheduled,
        sorties=sorties,
        deferred=deferred,
        total_km=sum(s.transit_km for s in sorties),
        total_minutes=sum(s.flight_minutes for s in sorties),
    )


def format_plan(plan: MissionPlan) -> str:
    """渲染成给 LLM / 前端阅读的文本。"""
    if not plan.tasks and not plan.deferred:
        return f"参考日期 {plan.reference_date}：{plan.line_name} 无需复飞的杆塔。"

    lines = [
        f"## 巡检任务计划（参考日期 {plan.reference_date}｜{plan.line_name}）",
        f"待飞杆塔 {len(plan.tasks)} 基，划分 {len(plan.sorties)} 个架次，"
        f"合计转场 {plan.total_km:.2f} km、飞行 {plan.total_minutes:.0f} min。",
        "",
        "### 任务清单（按优先级与时效排序）",
    ]
    for i, t in enumerate(plan.tasks, 1):
        overdue = f"，已超期 {t.overdue_days} 天" if t.overdue_days > 0 else ""
        lines.append(f"{i}. {t.asset_id} | 优先级 {t.priority} | 到期 {t.due_date}{overdue} | {t.reason}")

    lines += ["", "### 架次编排"]
    for s in plan.sorties:
        lines.append(
            f"- 架次 {s.index}：{len(s.asset_ids)} 基（{' → '.join(s.asset_ids)}），"
            f"转场 {s.transit_km:.2f} km，作业 {s.flight_minutes:.0f} min"
        )

    if plan.deferred:
        lines += ["", f"### 顺延（超出单日架次能力，共 {len(plan.deferred)} 基）"]
        for t in plan.deferred:
            lines.append(f"- {t.asset_id} | 优先级 {t.priority} | 到期 {t.due_date}")
    return "\n".join(lines)
