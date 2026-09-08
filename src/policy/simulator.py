"""内置作业仿真：让策略图在出勤之前先把一天飞一千遍

GaP 的自学习靠的是「在仿真里并行试跑不同的图，用成功率和吞吐挑图」。这里实现的是
同一件事的电力巡检版本：给定一张策略图，蒙特卡洛推演它在**扰动下的**执行结果。

### 这个仿真是什么，不是什么

不是飞行动力学仿真，没有六自由度模型、没有气动、没有航迹跟踪误差。
它是一个**作业时序与续航的随机模型**——只回答"按这张图排的架次，实际飞下来会不会中断、
一天能飞完多少基塔"。四个随机源，全部写在下面的常量里，口径公开可查：

    风况   在给定风速附近扰动，同时折损巡航地速与电池有效时长（顶风段更耗电）
    电池   每块电池的健康度不同，实际可用时长按健康系数打折
    悬停   单基作业时长有分散性（对光、复拍、遮挡）
    发现   一定概率发现疑似缺陷，需补拍，额外占用作业时间

架次内的失败分两级：**中断**是起飞前判断续航不够、主动返航（少飞几基，可接受）；
**备降/迫降**是已经出发、实际耗时超出可用续航（安全事件，直接否决该策略）。

已知的两处简化，读数时要记住：转场里程只统计架次**内部**的塔间航段，起降点到首基塔的往返
没有建模（计划层同样如此）；架次内各段转场按塔位数均摊，因为计划里只留了总里程。
两者都会让实际作业时间被低估，因此中断率是乐观估计。

这些系数是**工程量级的假设值，不是实测标定**，因此仿真产出的绝对数字没有外部效度；
它有效度的地方是**相对比较**——同一套随机数下，A 图和 B 图哪个更抗扰动。
自学习也只使用相对比较的结论。

### 公共随机数（CRN）

比较不同策略图时，同一 trial、同一基塔用的是同一组随机扰动
（种子由 seed/trial/asset_id 派生）。不这么做的话，两张图的差异会淹没在随机噪声里，
需要把 trial 数抬高一个量级才能分辨。
"""
from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass, field

from ..config import load_config
from .policy_graph import MissionPolicy, execute

# ---- 随机模型系数（工程假设，非实测标定；改这里等于改仿真口径） ----
WIND_SD_MPS = 2.0             # 风速扰动标准差
WIND_SPEED_PENALTY = 0.45     # 地速折损系数：往返顶/顺风抵消后的净损失比例
WIND_BATTERY_PENALTY = 0.35   # 续航折损系数：顶风段功率上升
BATTERY_HEALTH_RANGE = (0.88, 1.0)   # 电池健康度
HOVER_JITTER_SIGMA = 0.18     # 单基悬停时长对数正态扰动
FINDING_PROB = 0.12           # 单基发现疑似缺陷、需要补拍的概率
FINDING_EXTRA_RATIO = 0.5     # 补拍额外占用的悬停时长比例
BATTERY_SWAP_MINUTES = 5.0    # 架次间换电与转场准备


@dataclass
class SortieOutcome:
    index: int
    planned_towers: int
    completed_towers: int
    minutes: float
    battery_minutes: float
    aborted: bool
    incident: bool = False          # 实际耗时超出可用续航——备降/迫降，安全事件
    abort_reason: str = ""


@dataclass
class SimReport:
    policy_name: str
    trials: int
    due_towers: int            # 应飞（已扣除空域禁飞的不可飞塔位）
    scheduled_towers: int      # 本策略排进架次的塔位
    dropped_towers: int        # 被合规筛查剔除
    deferred_towers: int       # 超出单日能力顺延
    coverage: float            # 平均完成塔位 / 应飞塔位
    success_rate: float        # 全部架次无中断的 trial 占比
    sortie_abort_rate: float
    mean_completed: float
    towers_per_hour: float
    p95_sortie_minutes: float
    mean_battery_margin_min: float
    incident_rate: float = 0.0      # 发生备降/迫降的架次占比
    signature: str = ""
    notes: list[str] = field(default_factory=list)

    def rank_key(self) -> tuple:
        """排序口径：安全 → 完成比例 → 不中断率 → 吞吐，字典序比较。

        安全放第一位且不可交换：只要仿真里出现过备降/迫降，这张图就排在所有零事件的图之后，
        无论它的吞吐多好看。
        coverage 放第二位是为了堵住"少排点塔位就能刷满成功率"这条捷径——
        缩小作业面确实能把 success_rate 顶到 1.0，但 coverage 会同步下跌。
        """
        return (
            round(1.0 - self.incident_rate, 3),
            round(self.coverage, 3),
            round(self.success_rate, 3),
            round(self.towers_per_hour, 2),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def simulate(
    policy: MissionPolicy,
    reference_date: str,
    line_name: str | None = None,
    weather: dict | None = None,
    *,
    trials: int = 30,
    seed: int = 20260908,
) -> SimReport:
    """把一张策略图在扰动下跑 trials 遍，返回聚合指标。纯计算，不需要任何 API Key。"""
    plan = execute(policy, reference_date, line_name, weather)
    cfg = load_config()["mission"]
    fly = policy.node_by_skill("fly_sorties")
    rth_margin = fly.param("rth_margin_min") if fly else 0.0
    split = policy.node_by_skill("split_sorties")
    hover_planned = split.param("hover_minutes_per_tower") if split else cfg["hover_minutes_per_tower"]
    base_wind = float((weather or {}).get("wind_mps", 5.0))

    scheduled = sum(len(s.asset_ids) for s in plan.sorties)
    due = scheduled + len(plan.deferred)

    completed_per_trial: list[int] = []
    clean_trials = 0
    aborted_sorties = incident_sorties = total_sorties = 0
    sortie_minutes: list[float] = []
    margins: list[float] = []

    for trial in range(trials):
        env = random.Random(f"{seed}|env|{trial}")
        wind = max(0.0, env.gauss(base_wind, WIND_SD_MPS))
        speed_factor = max(0.35, 1.0 - WIND_SPEED_PENALTY * wind / cfg["cruise_speed_mps"])
        battery_factor = max(0.4, 1.0 - WIND_BATTERY_PENALTY * wind / cfg["cruise_speed_mps"])

        done = 0
        clean = True
        for sortie in plan.sorties:
            health = env.uniform(*BATTERY_HEALTH_RANGE)
            capacity = cfg["battery_minutes"] * health * battery_factor
            outcome = _fly_one(
                sortie, capacity, speed_factor, hover_planned, rth_margin,
                cruise_speed_mps=cfg["cruise_speed_mps"], seed=seed, trial=trial,
            )
            done += outcome.completed_towers
            total_sorties += 1
            sortie_minutes.append(outcome.minutes)
            margins.append(outcome.battery_minutes - outcome.minutes)
            if outcome.aborted:
                aborted_sorties += 1
                clean = False
            if outcome.incident:
                incident_sorties += 1
        completed_per_trial.append(done)
        clean_trials += int(clean)

    mean_completed = statistics.fmean(completed_per_trial) if completed_per_trial else 0.0
    total_minutes = (
        statistics.fmean(sortie_minutes) * len(plan.sorties)
        + BATTERY_SWAP_MINUTES * max(0, len(plan.sorties) - 1)
    ) if sortie_minutes else 0.0

    notes = []
    if plan.dropped:
        notes.append(f"{len(plan.dropped)} 基塔位被空域/气象筛查剔除，未计入应飞分母")
    if plan.deferred:
        notes.append(f"{len(plan.deferred)} 基塔位超出单日架次能力，顺延至次日")

    return SimReport(
        policy_name=policy.name,
        trials=trials,
        due_towers=due,
        scheduled_towers=scheduled,
        dropped_towers=len(plan.dropped),
        deferred_towers=len(plan.deferred),
        coverage=(mean_completed / due) if due else 0.0,
        success_rate=(clean_trials / trials) if trials else 0.0,
        sortie_abort_rate=(aborted_sorties / total_sorties) if total_sorties else 0.0,
        mean_completed=mean_completed,
        towers_per_hour=(mean_completed / total_minutes * 60.0) if total_minutes > 0 else 0.0,
        p95_sortie_minutes=_percentile(sortie_minutes, 0.95),
        mean_battery_margin_min=statistics.fmean(margins) if margins else 0.0,
        incident_rate=(incident_sorties / total_sorties) if total_sorties else 0.0,
        signature=policy.signature(),
        notes=notes,
    )


def _fly_one(sortie, capacity, speed_factor, hover_planned, rth_margin, *,
             cruise_speed_mps, seed, trial) -> SortieOutcome:
    """推演单个架次。每基塔起飞前先问一句：飞完它还回得来吗？回不来就现在返航。"""
    speed_m_per_min = cruise_speed_mps * 60.0 * speed_factor
    # 架次内各段转场按塔位数均摊（计划里只留了总里程，没留逐段明细）
    n = len(sortie.asset_ids)
    per_leg_min = (sortie.transit_km * 1000.0 / speed_m_per_min / max(1, n - 1)) if n > 1 else 0.0

    elapsed = 0.0
    completed = 0
    aborted = incident = False
    reason = ""
    for i, asset_id in enumerate(sortie.asset_ids):
        leg = per_leg_min if i > 0 else 0.0
        # 起飞去下一基之前用**计划值**做返航判断——机组决策时并不知道实际会悬停多久。
        # rth_margin 就是买这个信息差的保险费：留得多，白白返航；留得少，赌实际不超计划。
        if elapsed + leg + hover_planned + rth_margin > capacity:
            aborted = True
            reason = f"剩余续航不足以完成 {asset_id}（触发返航阈值 {rth_margin:.0f} min）"
            break

        rng = random.Random(f"{seed}|tower|{trial}|{asset_id}")
        hover = hover_planned * min(2.5, max(0.5, rng.lognormvariate(0.0, HOVER_JITTER_SIGMA)))
        if rng.random() < FINDING_PROB:
            hover += hover_planned * FINDING_EXTRA_RATIO

        if elapsed + leg + hover > capacity:
            # 赌输了：实际悬停超出计划，电量在空中耗尽——备降/迫降，计为安全事件
            aborted = incident = True
            reason = f"{asset_id} 实际作业耗时超出可用续航（余量 {capacity - elapsed - leg:.1f} min），发生备降"
            elapsed = capacity
            break
        elapsed += leg + hover
        completed += 1

    return SortieOutcome(sortie.index, n, completed, elapsed, capacity, aborted, incident, reason)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def format_sim(report: SimReport) -> str:
    lines = [
        f"## 策略仿真结果（{report.policy_name}｜{report.trials} 次蒙特卡洛）",
        f"图：{report.signature}",
        "",
        f"- 应飞塔位 {report.due_towers} 基（已排 {report.scheduled_towers}、"
        f"顺延 {report.deferred_towers}、合规剔除 {report.dropped_towers}）",
        f"- 完成率 coverage **{report.coverage:.1%}**（平均完成 {report.mean_completed:.1f} 基）",
        f"- 架次不中断率 **{report.success_rate:.1%}**，单架次中断率 {report.sortie_abort_rate:.1%}",
        f"- **备降/迫降率 {report.incident_rate:.1%}**（实际耗时超出可用续航；安全事件，非零即否决该策略）",
        f"- 吞吐 {report.towers_per_hour:.2f} 基/飞行小时，P95 单架次 {report.p95_sortie_minutes:.1f} min",
        f"- 平均剩余电量余量 {report.mean_battery_margin_min:.1f} min",
    ]
    lines += [f"- {n}" for n in report.notes]
    lines.append("")
    lines.append("> 仿真为作业时序与续航的随机模型（风况/电池健康/悬停分散/缺陷发现四个随机源），"
                 "系数为工程假设值而非实测标定，绝对数字仅供策略间横向比较。")
    return "\n".join(lines)
