"""巡检技能库（本仓库版的 MORSL）

参考 GaP（Graph-as-Policy, arXiv:2607.05369）的做法：先定义一个**带类型签名的技能库**，
再由图把技能连起来构成策略。技能库是策略搜索的封闭词表——图里只能出现这里登记过的技能，
这正是「生成的图可被静态校验」的前提。

每个技能声明四件事：
    kind    perception / planning / control，用于表达安全约束（control 之前必须有合规筛查）
    inputs  依赖的端口类型
    outputs 产出的端口类型
    params  可调参数及其取值域——自学习只在这个盒子里搜，越界即非法

技能的实现不在这里：本模块只描述**接口与取值域**，执行逻辑在 policy_graph.execute，
仿真逻辑在 simulator。把「图长什么样」和「图怎么跑」分开，图才能被序列化后交给别的执行器。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

# 端口类型：技能之间靠这些名字对接，validate 用它做静态类型检查
ASSET_SET = "AssetSet"
WEATHER_STATE = "WeatherState"
TASK_LIST = "TaskList"
CLEARED_TASKS = "ClearedTaskList"
ORDERED_TASKS = "OrderedTasks"
SORTIE_PLAN = "SortiePlan"
FLIGHT_OUTCOME = "FlightOutcome"

KINDS = ("perception", "planning", "control")


@dataclass(frozen=True)
class ParamSpec:
    """一个可调参数的取值域。low/high 是闭区间，自学习与人工改图都不得越界。"""

    default: float
    low: float
    high: float
    description: str

    def clamp(self, value: float) -> float:
        return max(self.low, min(self.high, float(value)))

    def contains(self, value: float) -> bool:
        return self.low - 1e-9 <= float(value) <= self.high + 1e-9


@dataclass(frozen=True)
class Skill:
    name: str
    kind: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    params: Mapping[str, ParamSpec]
    description: str

    def defaults(self) -> dict[str, float]:
        return {k: v.default for k, v in self.params.items()}


def _skill(name, kind, inputs, outputs, params, description) -> Skill:
    return Skill(name, kind, tuple(inputs), tuple(outputs),
                 MappingProxyType(dict(params)), description)


SKILLS: dict[str, Skill] = {s.name: s for s in [
    # ---------------- perception ----------------
    _skill(
        "sense_assets", "perception", (), (ASSET_SET,), {},
        "读取资产台账、缺陷案例与巡检历史，构成本次决策的世界状态",
    ),
    _skill(
        "sense_weather", "perception", (), (WEATHER_STATE,), {},
        "读取起飞前气象实况；缺失时下游合规筛查判为 unknown 而非放行",
    ),

    # ---------------- planning ----------------
    _skill(
        "collect_due_towers", "planning", (ASSET_SET,), (TASK_LIST,),
        {
            "lookahead_days": ParamSpec(
                0.0, 0.0, 30.0,
                "把未来 N 天内即将到期的塔位提前纳入本批次，用转场复用换取往返次数下降",
            ),
        },
        "按缺陷等级时效与例行间隔筛出待飞塔位，并按优先级、到期日排序",
    ),
    _skill(
        "screen_airspace", "planning", (TASK_LIST, WEATHER_STATE), (CLEARED_TASKS,),
        {
            "agl_m": ParamSpec(120.0, 30.0, 120.0, "计划作业真高（米）；压低可换取更多限高塔位放行"),
        },
        "逐基比对空域限制区与气象限值，剔除禁飞塔位，产出可飞清单",
    ),
    _skill(
        "order_route_nn", "planning", (CLEARED_TASKS,), (ORDERED_TASKS,), {},
        "同优先级组内按最近邻求访问顺序（快，但可能留下交叉航段）",
    ),
    _skill(
        "order_route_2opt", "planning", (CLEARED_TASKS,), (ORDERED_TASKS,), {},
        "同优先级组内最近邻 + 2-opt 去交叉，转场里程更短、单次装订耗时更高",
    ),
    _skill(
        "split_sorties", "planning", (ORDERED_TASKS,), (SORTIE_PLAN,),
        {
            "battery_reserve_pct": ParamSpec(
                0.0, 0.0, 40.0,
                "在配置的电池有效时长之上再留的安全余量百分比；越大越不容易中断，架次也越多",
            ),
            "max_towers_per_sortie": ParamSpec(
                12.0, 1.0, 12.0, "单架次塔位数硬上限，用于约束一次装订的作业规模",
            ),
            "hover_minutes_per_tower": ParamSpec(
                6.0, 3.0, 12.0, "为单基精细化巡视**预算**的悬停时长；实际时长在仿真中带扰动",
            ),
        },
        "按航程与续航预算把有序任务切成架次",
    ),

    # ---------------- control ----------------
    _skill(
        "fly_sorties", "control", (SORTIE_PLAN,), (FLIGHT_OUTCOME,),
        {
            "rth_margin_min": ParamSpec(
                3.0, 0.0, 10.0,
                "剩余续航低于该分钟数即中止本架次返航；调大更安全，但更容易漏飞尾部塔位",
            ),
        },
        "逐架次执行：转场 → 逐基悬停拍摄 → 返航换电；触发返航阈值则中止剩余塔位",
    ),
]}

SKILL_NAMES = tuple(SKILLS)


def skill(name: str) -> Skill:
    if name not in SKILLS:
        raise KeyError(f"技能库中没有 {name}；可用技能：{', '.join(SKILL_NAMES)}")
    return SKILLS[name]


def describe_library() -> str:
    """给 LLM 看的技能库说明——它只能用这些技能拼图。"""
    lines = ["## 巡检技能库（图节点只能取自下表）"]
    for kind in KINDS:
        lines.append(f"\n### {kind}")
        for s in SKILLS.values():
            if s.kind != kind:
                continue
            sig = f"({', '.join(s.inputs) or '—'}) -> ({', '.join(s.outputs)})"
            lines.append(f"- **{s.name}** {sig}\n  {s.description}")
            for pname, spec in s.params.items():
                lines.append(
                    f"  - 参数 `{pname}`: 默认 {spec.default}，取值域 [{spec.low}, {spec.high}]"
                    f"；{spec.description}"
                )
    return "\n".join(lines)
