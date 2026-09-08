"""策略图：静态校验、序列化、执行等价性"""
import json

import pytest

from src.policy.policy_graph import (
    MissionPolicy,
    Node,
    PolicyError,
    active_policy,
    default_policy,
    execute,
)
from src.policy.skills import SKILL_NAMES, skill

REF = "2025-10-01"
GOOD_WEATHER = {"condition": "晴", "wind_mps": 5.0, "gust_mps": 7.0,
                "visibility_km": 10.0, "temperature_c": 15.0}


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------

def test_baseline_policy_is_valid():
    assert default_policy().validate() == []


def test_unknown_skill_is_rejected():
    p = MissionPolicy("x", [Node("a", "teleport")], [])
    assert any("不存在的技能" in e for e in p.validate())


def test_undeclared_or_out_of_range_param_is_rejected():
    p = default_policy().with_params("x", split_sorties={"battery_reserve_pct": 999.0})
    assert any("越出取值域" in e for e in p.validate())
    p2 = default_policy().with_params("x", split_sorties={"warp_factor": 9.0})
    assert any("未声明的参数" in e for e in p2.validate())


def test_type_mismatch_between_ports_is_caught():
    """把 order_route 的输入接到 sense_weather 上——端口类型对不上。"""
    p = default_policy()
    p.edges = [("p1", "t1"), ("t1", "c1"), ("p2", "c1"),
               ("p2", "r1"), ("r1", "s1"), ("s1", "f1")]
    assert any("缺少输入" in e for e in p.validate())


def test_cycle_is_caught():
    p = default_policy()
    p.edges = list(p.edges) + [("f1", "t1")]
    assert any("环" in e for e in p.validate())


def test_control_node_without_upstream_clearance_is_rejected():
    """安全约束：没经过合规筛查就不许进 control 节点。"""
    p = MissionPolicy(
        "unsafe",
        [Node("p1", "sense_assets"), Node("t1", "collect_due_towers"),
         Node("r1", "order_route_2opt"), Node("s1", "split_sorties"),
         Node("f1", "fly_sorties")],
        [("p1", "t1"), ("t1", "r1"), ("r1", "s1"), ("s1", "f1")],
    )
    assert any("安全约束违规" in e for e in p.validate())


def test_check_raises_with_all_problems_at_once():
    p = MissionPolicy("x", [Node("a", "teleport"), Node("b", "warp")], [])
    with pytest.raises(PolicyError) as exc:
        p.check()
    assert "teleport" in str(exc.value) and "warp" in str(exc.value)


def test_exactly_one_terminal_required():
    p = default_policy()
    p.nodes = list(p.nodes) + [Node("f2", "fly_sorties")]
    p.edges = list(p.edges) + [("s1", "f2")]
    assert any("终端节点" in e for e in p.validate())


def test_every_skill_in_library_declares_a_known_kind():
    for name in SKILL_NAMES:
        s = skill(name)
        assert s.kind in ("perception", "planning", "control")
        assert s.outputs, f"{name} 不产出任何端口类型"


# --------------------------------------------------------------------------
# 序列化 —— 图要能脱离本进程存活
# --------------------------------------------------------------------------

def test_policy_survives_a_json_round_trip():
    original = default_policy().with_params(
        "tuned", split_sorties={"battery_reserve_pct": 15.0}, fly_sorties={"rth_margin_min": 4.0})
    restored = MissionPolicy.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.validate() == []
    assert restored.signature() == original.signature()


def test_committed_learned_policy_loads_and_validates():
    """仓库里固化的自学习产物必须是一张合法的图，否则 active_policy 会静默降级。"""
    p = active_policy()
    assert p.validate() == []


def test_with_params_does_not_mutate_the_source_policy():
    base = default_policy()
    before = base.signature()
    base.with_params("derived", split_sorties={"battery_reserve_pct": 30.0})
    assert base.signature() == before


# --------------------------------------------------------------------------
# 执行
# --------------------------------------------------------------------------

def test_baseline_execution_matches_plan_mission():
    """基线图的产出必须和既有 plan_mission 一致——否则"图即策略"是另起炉灶而非重构。"""
    from src.mission.planner import plan_mission

    line = "济南西郊 110kV 输电线路"
    ref_plan = plan_mission(REF, line)
    got = execute(default_policy(), REF, line, GOOD_WEATHER)
    assert [t.asset_id for t in got.tasks] == [t.asset_id for t in ref_plan.tasks]
    assert [s.asset_ids for s in got.sorties] == [s.asset_ids for s in ref_plan.sorties]


def test_execution_refuses_an_invalid_graph():
    bad = default_policy().with_params("bad", split_sorties={"battery_reserve_pct": -5.0})
    with pytest.raises(PolicyError):
        execute(bad, REF, None, GOOD_WEATHER)


def test_clearance_drops_forbidden_towers_from_the_plan():
    got = execute(default_policy(), REF, None, GOOD_WEATHER)
    assert got.dropped, "演示数据里存在禁飞塔位，应被合规筛查剔除"
    scheduled = {aid for s in got.sorties for aid in s.asset_ids}
    assert not (scheduled & set(got.dropped)), "禁飞塔位仍被排进了架次"


def test_max_towers_param_caps_every_sortie():
    p = default_policy().with_params("cap2", split_sorties={"max_towers_per_sortie": 2.0})
    got = execute(p, REF, None, GOOD_WEATHER)
    assert got.sorties
    assert all(len(s.asset_ids) <= 2 for s in got.sorties)


def test_battery_reserve_shrinks_the_per_sortie_budget():
    lean = execute(default_policy().with_params("r30", split_sorties={"battery_reserve_pct": 30.0}),
                   REF, None, GOOD_WEATHER)
    full = execute(default_policy(), REF, None, GOOD_WEATHER)
    assert max(s.flight_minutes for s in lean.sorties) <= max(s.flight_minutes for s in full.sorties)


def test_lookahead_never_shrinks_the_task_set():
    base = execute(default_policy(), REF, None, GOOD_WEATHER)
    ahead = execute(default_policy().with_params("la14", collect_due_towers={"lookahead_days": 14.0}),
                    REF, None, GOOD_WEATHER)
    assert len(ahead.tasks) + len(ahead.deferred) >= len(base.tasks) + len(base.deferred)


def test_two_route_skills_visit_the_same_towers():
    """换航线算法只应改变顺序，不应改变该飞哪些塔。"""
    a = execute(default_policy(), REF, None, GOOD_WEATHER)
    b = execute(default_policy().with_skill_swapped("order_route_2opt", "order_route_nn", "nn"),
                REF, None, GOOD_WEATHER)
    ids_a = {t.asset_id for t in a.tasks} | {t.asset_id for t in a.deferred}
    ids_b = {t.asset_id for t in b.tasks} | {t.asset_id for t in b.deferred}
    assert ids_a == ids_b
