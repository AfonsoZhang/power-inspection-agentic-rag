"""策略图：类型化、可静态校验、可序列化、可脱机执行

GaP 的核心主张是「图即策略」——LLM 只在**编译期**出现，把作业意图拼成一张图；
运行期由确定性执行器按图执行，不再需要模型。这么做换来三件本仓库原本没有的东西：

    1. **可校验**  图在跑之前就能查出类型不匹配、成环、以及安全约束违规
       （control 节点的祖先里必须有 screen_airspace——不许"先飞了再说"）。
    2. **可仿真**  图的参数集中在节点上，于是能把整张图丢进仿真里反复试跑（simulator.py）。
    3. **可脱机**  图能 to_dict / from_dict 存成 JSON，交给没有 API Key 的执行器直接跑。

与既有 mission 层的关系：本模块不重写规划算法，节点实现直接复用
planner / airspace / geo 里已被单测覆盖的函数，只是把「按什么顺序、用什么参数调用」
从写死的 plan_mission 里解放成一张可搜索的图。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from ..config import load_config
from ..ingestion.text_loader import load_assets
from ..mission.airspace import ClearanceReport, check_flight
from ..mission.geo import haversine_km, nearest_neighbor_order, optimize_route
from ..mission.planner import MissionTask, Sortie, collect_tasks
from .skills import FLIGHT_OUTCOME, SKILLS, Skill, skill

PRIORITY_ORDER = ("I", "II", "III", "routine")


# ---------------------------------------------------------------------------
# 图结构
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: str
    skill: str
    params: dict[str, float] = field(default_factory=dict)

    def spec(self) -> Skill:
        return skill(self.skill)

    def param(self, name: str) -> float:
        """取参数值：节点上没写就用技能默认值。"""
        if name in self.params:
            return float(self.params[name])
        return self.spec().params[name].default


class PolicyError(ValueError):
    """图非法。带上全部问题，而不是碰到第一个就抛——改图的人需要一次看全。"""


@dataclass
class MissionPolicy:
    name: str
    nodes: list[Node]
    edges: list[tuple[str, str]]
    notes: str = ""

    # -------------------- 校验 --------------------
    def validate(self) -> list[str]:
        errors: list[str] = []
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            errors.append("存在重复的节点 id")
        index = {n.id: n for n in self.nodes}

        for n in self.nodes:
            if n.skill not in SKILLS:
                errors.append(f"节点 {n.id} 引用了技能库中不存在的技能 {n.skill}")
                continue
            spec = n.spec()
            for pname, pvalue in n.params.items():
                if pname not in spec.params:
                    errors.append(f"节点 {n.id} 有技能 {n.skill} 未声明的参数 {pname}")
                elif not spec.params[pname].contains(pvalue):
                    pspec = spec.params[pname]
                    errors.append(
                        f"节点 {n.id} 的 {pname}={pvalue} 越出取值域 [{pspec.low}, {pspec.high}]"
                    )

        for src, dst in self.edges:
            if src not in index:
                errors.append(f"边 {src}->{dst} 的起点不存在")
            if dst not in index:
                errors.append(f"边 {src}->{dst} 的终点不存在")
        if errors:
            return errors  # 结构都不完整，后面的类型/环检查没有意义

        order = self._topo_ids()
        if order is None:
            return ["图中存在环，无法确定执行顺序"]

        # 类型检查：每个节点的 inputs 必须全部由前驱的 outputs 提供
        preds: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for src, dst in self.edges:
            preds[dst].append(src)
        for nid in order:
            node = index[nid]
            available = {t for p in preds[nid] for t in index[p].spec().outputs}
            missing = [t for t in node.spec().inputs if t not in available]
            if missing:
                errors.append(f"节点 {nid}({node.skill}) 缺少输入 {', '.join(missing)}")

        # 安全约束：任何 control 节点的祖先里必须有合规筛查
        ancestors = self._ancestors(preds, order)
        for n in self.nodes:
            if n.spec().kind != "control":
                continue
            if not any(index[a].skill == "screen_airspace" for a in ancestors[n.id]):
                errors.append(
                    f"安全约束违规：control 节点 {n.id} 的上游没有 screen_airspace，"
                    "未经合规筛查不得进入执行"
                )

        terminals = [n for n in self.nodes if FLIGHT_OUTCOME in n.spec().outputs]
        if len(terminals) != 1:
            errors.append(f"图必须恰好有一个产出 {FLIGHT_OUTCOME} 的终端节点，当前有 {len(terminals)} 个")
        return errors

    def check(self) -> MissionPolicy:
        errors = self.validate()
        if errors:
            raise PolicyError("策略图校验未通过：\n- " + "\n- ".join(errors))
        return self

    def _topo_ids(self) -> list[str] | None:
        indeg = {n.id: 0 for n in self.nodes}
        succ: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for src, dst in self.edges:
            indeg[dst] += 1
            succ[src].append(dst)
        queue = sorted(nid for nid, d in indeg.items() if d == 0)
        order: list[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for nxt in succ[nid]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
            queue.sort()
        return order if len(order) == len(self.nodes) else None

    @staticmethod
    def _ancestors(preds: dict[str, list[str]], order: list[str]) -> dict[str, set[str]]:
        anc: dict[str, set[str]] = {}
        for nid in order:
            acc: set[str] = set()
            for p in preds[nid]:
                acc.add(p)
                acc |= anc.get(p, set())
            anc[nid] = acc
        return anc

    def topo_nodes(self) -> list[Node]:
        order = self._topo_ids()
        if order is None:
            raise PolicyError("图中存在环，无法确定执行顺序")
        index = {n.id: n for n in self.nodes}
        return [index[nid] for nid in order]

    def node_by_skill(self, name: str) -> Node | None:
        return next((n for n in self.nodes if n.skill == name), None)

    def signature(self) -> str:
        """人可读的图指纹，用于在自学习结果里区分变体。"""
        parts = []
        for n in self.topo_nodes():
            ps = ",".join(f"{k}={v:g}" for k, v in sorted(n.params.items()))
            parts.append(f"{n.skill}({ps})" if ps else n.skill)
        return " → ".join(parts)

    # -------------------- 序列化 --------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "notes": self.notes,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [list(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> MissionPolicy:
        return cls(
            name=data.get("name", "unnamed"),
            nodes=[Node(n["id"], n["skill"], dict(n.get("params") or {})) for n in data["nodes"]],
            edges=[(e[0], e[1]) for e in data.get("edges", [])],
            notes=data.get("notes", ""),
        )

    def with_params(self, name: str, **overrides: dict[str, float]) -> MissionPolicy:
        """派生一个只改参数的新图（自学习靠它生成变体，原图不被修改）。"""
        nodes = [Node(n.id, n.skill, dict(n.params)) for n in self.nodes]
        by_skill = {n.skill: n for n in nodes}
        for skill_name, params in overrides.items():
            if skill_name in by_skill:
                by_skill[skill_name].params.update(params)
        return MissionPolicy(name, nodes, list(self.edges), self.notes)

    def with_skill_swapped(self, old: str, new: str, name: str) -> MissionPolicy:
        """把某个节点换成同签名的另一技能（结构变体，例如最近邻 ↔ 2-opt）。"""
        nodes = [
            Node(n.id, new if n.skill == old else n.skill, dict(n.params))
            for n in self.nodes
        ]
        return MissionPolicy(name, nodes, list(self.edges), self.notes)


def default_policy() -> MissionPolicy:
    """基线策略：等价于当前 plan_mission 的行为，作为自学习的对照组。"""
    return MissionPolicy(
        name="baseline",
        nodes=[
            Node("p1", "sense_assets"),
            Node("p2", "sense_weather"),
            Node("t1", "collect_due_towers"),
            Node("c1", "screen_airspace"),
            Node("r1", "order_route_2opt"),
            Node("s1", "split_sorties"),
            Node("f1", "fly_sorties"),
        ],
        edges=[("p1", "t1"), ("t1", "c1"), ("p2", "c1"),
               ("c1", "r1"), ("r1", "s1"), ("s1", "f1")],
        notes="与 mission.plan_mission 行为一致的基线：全高度作业、零额外余量、2-opt 航线",
    )


def active_policy() -> MissionPolicy:
    """当前在用的策略图：磁盘上有自学习产物就用它，否则用基线。

    这一步就是 GaP 说的"脱离 agent 执行"——图是一份 JSON，谁都能加载、校验、执行，
    不需要模型、不需要 API Key。图不合法则拒绝加载并回落基线，绝不带病上天。
    """
    path = load_config()["_paths"].get("learned_policy")
    if not path or not path.exists():
        return default_policy()
    try:
        policy = MissionPolicy.from_dict(json.loads(path.read_text(encoding="utf-8")))
        policy.check()
        return policy
    except (OSError, ValueError, KeyError, PolicyError):
        return default_policy()


# ---------------------------------------------------------------------------
# 执行：按拓扑序调用真实实现，全程不碰 LLM
# ---------------------------------------------------------------------------

@dataclass
class PolicyExecution:
    policy_name: str
    reference_date: str
    line_name: str
    tasks: list[MissionTask]
    sorties: list[Sortie]
    deferred: list[MissionTask]
    dropped: list[str]                 # 被合规筛查剔除的塔位
    clearance: ClearanceReport | None
    total_km: float
    planned_minutes: float

    def to_dict(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "reference_date": self.reference_date,
            "line_name": self.line_name,
            "tasks": [t.to_dict() for t in self.tasks],
            "sorties": [s.to_dict() for s in self.sorties],
            "deferred": [t.to_dict() for t in self.deferred],
            "dropped": self.dropped,
            "clearance": self.clearance.to_dict() if self.clearance else None,
            "total_km": round(self.total_km, 3),
            "planned_minutes": round(self.planned_minutes, 1),
        }


def execute(
    policy: MissionPolicy,
    reference_date: str,
    line_name: str | None = None,
    weather: dict | None = None,
) -> PolicyExecution:
    """按图执行一次规划。图非法直接抛错——不许带着违规的图上天。"""
    policy.check()
    cfg = load_config()["mission"]
    state: dict[str, object] = {}

    for node in policy.topo_nodes():
        kind = node.skill

        if kind == "sense_assets":
            state["assets"] = load_assets()

        elif kind == "sense_weather":
            state["weather"] = weather

        elif kind == "collect_due_towers":
            horizon = (
                datetime.strptime(reference_date, "%Y-%m-%d").date()
                + timedelta(days=int(node.param("lookahead_days")))
            ).isoformat()
            state["tasks"] = collect_tasks(horizon, line_name)

        elif kind == "screen_airspace":
            tasks: list[MissionTask] = state.get("tasks") or []
            agl = node.param("agl_m")
            report = check_flight([t.asset_id for t in tasks], agl_m=agl,
                                  weather=state.get("weather")) if tasks else None
            blocked = {a.asset_id for a in report.assets if a.verdict == "forbidden"} if report else set()
            state["clearance"] = report
            state["dropped"] = sorted(blocked)
            state["cleared"] = [t for t in tasks if t.asset_id not in blocked]

        elif kind in ("order_route_nn", "order_route_2opt"):
            state["ordered"] = _order_within_priority(state.get("cleared") or [], use_two_opt=(kind.endswith("2opt")))

        elif kind == "split_sorties":
            sorties, deferred = split_sorties(
                state.get("ordered") or [],
                cruise_speed_mps=cfg["cruise_speed_mps"],
                hover_minutes=node.param("hover_minutes_per_tower"),
                battery_minutes=cfg["battery_minutes"],
                reserve_pct=node.param("battery_reserve_pct"),
                max_towers=int(node.param("max_towers_per_sortie")),
                max_sorties=cfg["max_sorties_per_day"],
            )
            state["sorties"], state["deferred"] = sorties, deferred

        elif kind == "fly_sorties":
            pass  # 真实执行属于机载端；本仓库只在 simulator 里推演它

    sorties: list[Sortie] = state.get("sorties") or []
    scheduled = {aid for s in sorties for aid in s.asset_ids}
    ordered: list[MissionTask] = state.get("ordered") or []
    return PolicyExecution(
        policy_name=policy.name,
        reference_date=reference_date,
        line_name=line_name or "全部线路",
        tasks=[t for t in ordered if t.asset_id in scheduled],
        sorties=sorties,
        deferred=state.get("deferred") or [],
        dropped=state.get("dropped") or [],
        clearance=state.get("clearance"),
        total_km=sum(s.transit_km for s in sorties),
        planned_minutes=sum(s.flight_minutes for s in sorties),
    )


def _order_within_priority(tasks: list[MissionTask], *, use_two_opt: bool) -> list[MissionTask]:
    """航线优化只在同优先级组内做——时效是硬约束，省航程是软目标。"""
    ordered: list[MissionTask] = []
    for priority in PRIORITY_ORDER:
        group = [t for t in tasks if t.priority == priority]
        if not group:
            continue
        pts = [(t.lat, t.lon) for t in group]
        idx = optimize_route(pts, start=0)[0] if use_two_opt else nearest_neighbor_order(pts, start=0)
        ordered.extend(group[i] for i in idx)
    return ordered


def split_sorties(
    tasks: list[MissionTask],
    *,
    cruise_speed_mps: float,
    hover_minutes: float,
    battery_minutes: float,
    reserve_pct: float,
    max_towers: int,
    max_sorties: int,
) -> tuple[list[Sortie], list[MissionTask]]:
    """参数化版的架次切分：在 mission.split_into_sorties 之上多了安全余量与单架次塔位上限。"""
    budget = battery_minutes * (1.0 - reserve_pct / 100.0)
    speed_m_per_min = cruise_speed_mps * 60.0

    sorties: list[Sortie] = []
    current = Sortie(index=1)
    prev: MissionTask | None = None

    for task in tasks:
        transit_km = haversine_km((prev.lat, prev.lon), (task.lat, task.lon)) if prev else 0.0
        transit_min = transit_km * 1000.0 / speed_m_per_min
        needed = current.flight_minutes + transit_min + hover_minutes
        over_budget = current.asset_ids and needed > budget
        over_count = len(current.asset_ids) >= max_towers

        if over_budget or over_count:
            sorties.append(current)
            current = Sortie(index=len(sorties) + 1)
            transit_km, transit_min = 0.0, 0.0   # 新架次从起降点重新起飞
            needed = hover_minutes

        current.asset_ids.append(task.asset_id)
        current.transit_km += transit_km
        current.hover_minutes += hover_minutes
        current.flight_minutes = needed
        prev = task

    if current.asset_ids:
        sorties.append(current)

    if len(sorties) <= max_sorties:
        return sorties, []
    kept = sorties[:max_sorties]
    scheduled = {aid for s in kept for aid in s.asset_ids}
    return kept, [t for t in tasks if t.asset_id not in scheduled]
