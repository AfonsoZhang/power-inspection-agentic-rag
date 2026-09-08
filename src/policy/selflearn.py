"""自学习：在仿真里试跑图的变体，挑出比基线更抗扰动的那张

这是 GaP 闭环里 LLM **不参与**的那一半。搜索空间完全由技能库的取值域界定，
评价完全由 simulator 给出，因此整个过程确定性、可复现、免 API Key——
CI 里就是这么跑的。

搜什么：
    结构   航线技能二选一（最近邻 / 2-opt）
    参数   电池安全余量、单架次塔位上限、返航阈值、提前纳入天数

**不搜什么，以及为什么**：
    hover_minutes_per_tower 不在搜索空间里。仿真中「实际悬停」是以计划悬停为基准扰动的，
    调小计划值会让实际作业量跟着缩水——那是在骗仿真器，不是在改进策略。
    单基作业需要多久由拍摄科目决定，属于输入而非可优化量。
    agl_m 同理不搜：本仓库的空域数据只区分能飞/不能飞，压低真高在仿真里不产生任何代价，
    搜它只会得到"越低越好"这种没有信息量的结论。

搜索是穷举网格且**串行**执行的（GaP 原文用并行 rehearsal）。这里的变体规模是百量级、
单次仿真几十毫秒，串行足够；写成并行只会引入无谓的复杂度。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from .policy_graph import MissionPolicy, default_policy
from .simulator import SimReport, simulate

DEFAULT_GRID = {
    "route": ("order_route_2opt", "order_route_nn"),
    "battery_reserve_pct": (0.0, 10.0, 20.0, 30.0),
    "max_towers_per_sortie": (2.0, 3.0, 4.0, 6.0, 12.0),
    "rth_margin_min": (0.0, 3.0, 6.0),
    "lookahead_days": (0.0, 7.0),
}


@dataclass
class LearningReport:
    reference_date: str
    line_name: str
    trials: int
    evaluated: int
    baseline: SimReport
    best: SimReport
    best_policy: MissionPolicy
    top: list[SimReport] = field(default_factory=list)
    unsafe_rejected: int = 0
    unsafe_best: SimReport | None = None   # 被安全判据否决掉的最高完成率候选

    @property
    def improved(self) -> bool:
        return self.best.rank_key() > self.baseline.rank_key()

    def to_dict(self) -> dict:
        return {
            "reference_date": self.reference_date,
            "line_name": self.line_name,
            "trials": self.trials,
            "evaluated": self.evaluated,
            "improved": self.improved,
            "baseline": self.baseline.to_dict(),
            "best": self.best.to_dict(),
            "best_policy": self.best_policy.to_dict(),
            "top": [r.to_dict() for r in self.top],
            "unsafe_rejected": self.unsafe_rejected,
            "unsafe_best": self.unsafe_best.to_dict() if self.unsafe_best else None,
        }


def variants(base: MissionPolicy, grid: dict | None = None) -> list[MissionPolicy]:
    """按网格派生候选图。基线本身也在候选里（网格含其默认取值时）。"""
    g = {**DEFAULT_GRID, **(grid or {})}
    out: list[MissionPolicy] = []
    for route, reserve, cap, rth, look in product(
        g["route"], g["battery_reserve_pct"], g["max_towers_per_sortie"],
        g["rth_margin_min"], g["lookahead_days"],
    ):
        tag = f"{route.removeprefix('order_route_')}|res{reserve:g}|cap{cap:g}|rth{rth:g}|la{look:g}"
        policy = base.with_skill_swapped("order_route_2opt", route, tag)
        policy = policy.with_skill_swapped("order_route_nn", route, tag)
        policy = policy.with_params(
            tag,
            split_sorties={"battery_reserve_pct": reserve, "max_towers_per_sortie": cap},
            fly_sorties={"rth_margin_min": rth},
            collect_due_towers={"lookahead_days": look},
        )
        if not policy.validate():
            out.append(policy)
    return out


def search(
    reference_date: str,
    line_name: str | None = None,
    weather: dict | None = None,
    *,
    trials: int = 24,
    seed: int = 20260908,
    grid: dict | None = None,
    top_k: int = 5,
) -> LearningReport:
    """在仿真里评估全部候选图，返回基线 vs 最优的对照。"""
    base = default_policy()
    baseline = simulate(base, reference_date, line_name, weather, trials=trials, seed=seed)

    scored: list[SimReport] = []
    best_policy = base
    best_report = baseline
    for policy in variants(base, grid):
        # 所有变体共用同一 seed：同一 trial、同一基塔拿到同一组扰动（公共随机数），
        # 图之间的差异才不会被随机噪声淹没。
        report = simulate(policy, reference_date, line_name, weather, trials=trials, seed=seed)
        scored.append(report)
        if report.rank_key() > best_report.rank_key():
            best_report, best_policy = report, policy

    scored.sort(key=lambda r: r.rank_key(), reverse=True)
    unsafe = [r for r in scored if r.incident_rate > 0]
    unsafe_best = max(unsafe, key=lambda r: r.coverage, default=None)
    return LearningReport(
        reference_date=reference_date,
        line_name=line_name or "全部线路",
        trials=trials,
        evaluated=len(scored),
        baseline=baseline,
        best=best_report,
        best_policy=best_policy,
        top=scored[:top_k],
        unsafe_rejected=len(unsafe),
        unsafe_best=unsafe_best,
    )


def format_learning(report: LearningReport) -> str:
    b, x = report.baseline, report.best
    lines = [
        f"## 策略自学习结果（{report.line_name}｜参考日期 {report.reference_date}）",
        f"候选图 {report.evaluated} 张，各跑 {report.trials} 次蒙特卡洛（公共随机数）。",
        "",
        "| 指标 | 基线 | 最优 |",
        "| --- | --- | --- |",
        f"| 备降/迫降率 | {b.incident_rate:.1%} | {x.incident_rate:.1%} |",
        f"| 完成率 coverage | {b.coverage:.1%} | {x.coverage:.1%} |",
        f"| 架次不中断率 | {b.success_rate:.1%} | {x.success_rate:.1%} |",
        f"| 吞吐（基/飞行小时） | {b.towers_per_hour:.2f} | {x.towers_per_hour:.2f} |",
        f"| 平均完成塔位 | {b.mean_completed:.1f} | {x.mean_completed:.1f} |",
        "",
        f"**最优图**：{x.signature}",
    ]
    if not report.improved:
        lines.append("\n基线已是当前搜索空间内的最优，无需改图。")

    u = report.unsafe_best
    if u is not None:
        lines += [
            "",
            f"### 安全判据否决了 {report.unsafe_rejected} 张图",
            f"其中完成率最高的是 `{u.policy_name}`：coverage {u.coverage:.1%}"
            f"（高于最优图的 {x.coverage:.1%}），但备降/迫降率 {u.incident_rate:.1%}。",
            "这正是把安全放在排序第一位的意义——只按吞吐挑图会挑中它。",
        ]
    if report.top:
        lines += ["", "### 前几名"]
        for i, r in enumerate(report.top, 1):
            lines.append(
                f"{i}. `{r.policy_name}` — 迫降 {r.incident_rate:.1%}，"
                f"coverage {r.coverage:.1%}，不中断 {r.success_rate:.1%}，"
                f"吞吐 {r.towers_per_hour:.2f}"
            )
    lines += ["", "> 结论只在本仿真口径下成立（系数为工程假设值），"
                  "用于说明「图可被搜索与验证」这一机制，不构成真实作业参数建议。"]
    return "\n".join(lines)
