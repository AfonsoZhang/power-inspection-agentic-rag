"""作业仿真与自学习：可复现性、指标自洽、排序口径不被吞吐钻空子"""
import pytest

from src.policy.policy_graph import default_policy
from src.policy.selflearn import search, variants
from src.policy.simulator import SimReport, format_sim, simulate

REF = "2025-10-01"
LINE = "济南西郊 110kV 输电线路"
W = {"condition": "晴", "wind_mps": 5.0, "gust_mps": 7.0,
     "visibility_km": 10.0, "temperature_c": 15.0}
TINY_GRID = {
    "route": ("order_route_2opt",),
    "battery_reserve_pct": (0.0, 20.0),
    "max_towers_per_sortie": (2.0, 12.0),
    "rth_margin_min": (0.0, 3.0),
    "lookahead_days": (0.0,),
}


# --------------------------------------------------------------------------
# 可复现性 —— 没有它，仿真结论就没法写进 README
# --------------------------------------------------------------------------

def test_same_seed_gives_identical_numbers():
    a = simulate(default_policy(), REF, LINE, W, trials=8, seed=7)
    b = simulate(default_policy(), REF, LINE, W, trials=8, seed=7)
    assert a.to_dict() == b.to_dict()


def test_different_seed_actually_changes_the_draw():
    a = simulate(default_policy(), REF, LINE, W, trials=8, seed=7)
    b = simulate(default_policy(), REF, LINE, W, trials=8, seed=8)
    assert (a.mean_completed, a.p95_sortie_minutes) != (b.mean_completed, b.p95_sortie_minutes)


def test_simulation_needs_no_api_key():
    """整条仿真链是纯计算——CI 正是在没有任何 key 的环境里跑它。"""
    assert simulate(default_policy(), REF, LINE, W, trials=5).trials == 5


# --------------------------------------------------------------------------
# 指标自洽
# --------------------------------------------------------------------------

def test_due_towers_equals_scheduled_plus_deferred():
    r = simulate(default_policy(), REF, None, W, trials=5)
    assert r.due_towers == r.scheduled_towers + r.deferred_towers


def test_completed_never_exceeds_scheduled():
    r = simulate(default_policy(), REF, None, W, trials=10)
    assert 0.0 <= r.mean_completed <= r.scheduled_towers
    assert 0.0 <= r.coverage <= 1.0


@pytest.mark.parametrize("rate", ["success_rate", "sortie_abort_rate", "incident_rate"])
def test_rates_stay_within_zero_and_one(rate):
    r = simulate(default_policy(), REF, None, W, trials=10)
    assert 0.0 <= getattr(r, rate) <= 1.0


def test_stronger_wind_never_helps():
    calm = simulate(default_policy(), REF, None, {**W, "wind_mps": 2.0}, trials=12)
    gale = simulate(default_policy(), REF, None, {**W, "wind_mps": 11.0}, trials=12)
    assert gale.mean_completed <= calm.mean_completed


def test_bigger_return_margin_never_increases_incidents():
    """返航阈值是安全余量：调大它不应该让备降变多。"""
    lo = simulate(default_policy().with_params("rth0", fly_sorties={"rth_margin_min": 0.0}),
                  REF, None, W, trials=20)
    hi = simulate(default_policy().with_params("rth8", fly_sorties={"rth_margin_min": 8.0}),
                  REF, None, W, trials=20)
    assert hi.incident_rate <= lo.incident_rate


def test_empty_plan_reports_zeroes_instead_of_dividing_by_zero():
    r = simulate(default_policy(), REF, "不存在的 110kV 线路", W, trials=3)
    assert r.due_towers == 0 and r.coverage == 0.0 and r.towers_per_hour == 0.0


def test_format_sim_carries_the_model_disclaimer():
    text = format_sim(simulate(default_policy(), REF, LINE, W, trials=3))
    assert "工程假设值" in text and "备降" in text


# --------------------------------------------------------------------------
# 排序口径
# --------------------------------------------------------------------------

def _report(**kw) -> SimReport:
    base = {"policy_name": "x", "trials": 10, "due_towers": 10, "scheduled_towers": 10,
            "dropped_towers": 0, "deferred_towers": 0, "coverage": 0.5, "success_rate": 0.5,
            "sortie_abort_rate": 0.1, "mean_completed": 5.0, "towers_per_hour": 6.0,
            "p95_sortie_minutes": 20.0, "mean_battery_margin_min": 5.0, "incident_rate": 0.0}
    return SimReport(**{**base, **kw})


def test_any_incident_loses_to_a_clean_policy_however_good_its_numbers():
    unsafe = _report(coverage=0.99, success_rate=1.0, towers_per_hour=99.0, incident_rate=0.01)
    safe = _report(coverage=0.10, success_rate=0.1, towers_per_hour=1.0, incident_rate=0.0)
    assert safe.rank_key() > unsafe.rank_key()


def test_shrinking_the_workload_cannot_buy_a_better_rank():
    """少排塔位能把 success_rate 刷到 1.0，但 coverage 会掉——排序必须看穿这一招。"""
    gamed = _report(coverage=0.20, success_rate=1.0, towers_per_hour=9.0)
    honest = _report(coverage=0.60, success_rate=0.3, towers_per_hour=6.0)
    assert honest.rank_key() > gamed.rank_key()


# --------------------------------------------------------------------------
# 自学习
# --------------------------------------------------------------------------

def test_every_generated_variant_is_a_valid_graph():
    for p in variants(default_policy(), TINY_GRID):
        assert p.validate() == [], f"{p.name} 生成了非法图"


def test_variant_count_matches_the_grid():
    assert len(variants(default_policy(), TINY_GRID)) == 1 * 2 * 2 * 2 * 1


def test_search_never_returns_something_worse_than_the_baseline():
    r = search(REF, LINE, W, trials=6, grid=TINY_GRID)
    assert r.best.rank_key() >= r.baseline.rank_key()
    assert r.best_policy.validate() == []


def test_search_is_reproducible():
    a = search(REF, LINE, W, trials=6, seed=11, grid=TINY_GRID)
    b = search(REF, LINE, W, trials=6, seed=11, grid=TINY_GRID)
    assert a.best.policy_name == b.best.policy_name
    assert a.to_dict() == b.to_dict()


def test_search_result_never_carries_a_safety_incident():
    r = search(REF, None, W, trials=8, grid=TINY_GRID)
    assert r.best.incident_rate == 0.0, "被选中的图不允许出现备降/迫降"
