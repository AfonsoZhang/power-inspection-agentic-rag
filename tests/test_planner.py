"""复飞任务规划：时效判定、闭环判据、优先级与架次切分"""
from datetime import datetime, timedelta

import pytest

from src.config import load_config
from src.mission.planner import (
    SEVERITY_RANK,
    MissionTask,
    _open_defects,
    collect_tasks,
    format_plan,
    plan_mission,
    split_into_sorties,
)

REF = "2025-10-01"


def _d(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


# --------------------------------------------------------------------------
# 闭环判据
# --------------------------------------------------------------------------

def test_defect_closed_when_later_inspection_no_longer_reports_it():
    cases = [{"case_id": "C1", "asset_id": "A", "severity": "II", "discovery_date": "2025-01-01",
              "defect_type": "螺栓松动"}]
    history = [{"asset_id": "A", "date": "2025-02-01", "findings": []}]
    assert _open_defects(cases, history) == {}


def test_defect_stays_open_when_reconfirmed_by_later_inspection():
    cases = [{"case_id": "C1", "asset_id": "A", "severity": "II", "discovery_date": "2025-01-01",
              "defect_type": "螺栓松动"}]
    history = [{"asset_id": "A", "date": "2025-02-01", "findings": ["C1"]}]
    assert list(_open_defects(cases, history)) == ["A"]


def test_defect_stays_open_when_no_later_inspection():
    cases = [{"case_id": "C1", "asset_id": "A", "severity": "I", "discovery_date": "2025-01-01",
              "defect_type": "导线断股"}]
    history = [{"asset_id": "A", "date": "2024-12-01", "findings": []}]
    assert list(_open_defects(cases, history)) == ["A"]


# --------------------------------------------------------------------------
# 任务识别与排序
# --------------------------------------------------------------------------

def test_collect_tasks_sorted_by_priority_then_due_date():
    tasks = collect_tasks(REF)
    ranks = [SEVERITY_RANK.get(t.priority, 3) for t in tasks]
    assert ranks == sorted(ranks), "任务未按优先级排序"
    for a, b in zip(tasks, tasks[1:], strict=False):
        if a.priority == b.priority:
            assert a.due_date <= b.due_date, "同优先级未按到期日升序"


def test_defect_recheck_due_date_follows_regulation_window():
    recheck = load_config()["mission"]["recheck_days_by_severity"]
    for t in collect_tasks(REF):
        if t.task_type != "defect_recheck":
            continue
        # 到期日 = 发现日 + 该等级时效；反推出的间隔必须落在配置表里
        assert t.priority in recheck
        assert t.overdue_days == max(0, (_d(REF) - _d(t.due_date)).days)


def test_routine_tasks_only_when_interval_exceeded():
    cfg = load_config()["mission"]
    for t in collect_tasks(REF):
        if t.task_type == "routine":
            assert _d(t.due_date) <= _d(REF)
    assert cfg["routine_interval_days"] > 0


def test_line_filter_returns_only_that_line():
    line = "青岛沿海 110kV 输电线路"
    tasks = collect_tasks(REF, line)
    assert tasks, "该线路应有待飞塔位"
    assert {t.line_name for t in tasks} == {line}


# --------------------------------------------------------------------------
# 架次切分
# --------------------------------------------------------------------------

def _fake_tasks(n, spacing_deg=0.004):
    return [
        MissionTask(asset_id=f"T{i:03d}", line_name="L", lat=36.0 + i * spacing_deg, lon=116.0,
                    task_type="routine", priority="routine", reason="", due_date=REF, overdue_days=0)
        for i in range(n)
    ]


def test_every_sortie_fits_within_battery_budget():
    cfg = load_config()["mission"]
    sorties, _ = split_into_sorties(_fake_tasks(20))
    assert sorties
    for s in sorties:
        assert s.flight_minutes <= cfg["battery_minutes"] + 1e-9, f"架次 {s.index} 超出电池续航"


def test_sorties_partition_tasks_without_loss_or_duplication():
    tasks = _fake_tasks(12)
    sorties, deferred = split_into_sorties(tasks)
    scheduled = [aid for s in sorties for aid in s.asset_ids]
    assert len(scheduled) == len(set(scheduled)), "有塔位被排进多个架次"
    assert set(scheduled) | {t.asset_id for t in deferred} == {t.asset_id for t in tasks}


def test_tasks_beyond_daily_capacity_are_deferred_not_dropped():
    cfg = load_config()["mission"]
    # 每基 hover 6 min、电池 28 min => 单架次至多 4 基；给足够多的塔位逼出顺延
    tasks = _fake_tasks(cfg["max_sorties_per_day"] * 6 + 10)
    sorties, deferred = split_into_sorties(tasks)
    assert len(sorties) == cfg["max_sorties_per_day"]
    assert deferred, "超出单日能力的塔位应进入顺延而不是被丢弃"


def test_single_task_forms_one_sortie():
    sorties, deferred = split_into_sorties(_fake_tasks(1))
    assert len(sorties) == 1 and sorties[0].asset_ids == ["T000"] and not deferred


def test_empty_task_list_yields_no_sortie():
    assert split_into_sorties([]) == ([], [])


# --------------------------------------------------------------------------
# 端到端
# --------------------------------------------------------------------------

def test_plan_mission_priority_groups_stay_in_order():
    plan = plan_mission(REF)
    ranks = [SEVERITY_RANK.get(t.priority, 3) for t in plan.tasks]
    assert ranks == sorted(ranks), "航线优化把高优先级塔位排到了低优先级之后"


def test_plan_mission_reports_consistent_totals():
    plan = plan_mission(REF)
    assert plan.total_km == pytest.approx(sum(s.transit_km for s in plan.sorties))
    assert plan.total_minutes == pytest.approx(sum(s.flight_minutes for s in plan.sorties))
    assert len(plan.tasks) == sum(len(s.asset_ids) for s in plan.sorties)


def test_format_plan_mentions_every_scheduled_asset():
    plan = plan_mission(REF, "济南西郊 110kV 输电线路")
    text = format_plan(plan)
    for t in plan.tasks:
        assert t.asset_id in text


def test_plan_for_unknown_line_is_empty_not_error():
    plan = plan_mission(REF, "不存在的 110kV 线路")
    assert plan.tasks == [] and plan.sorties == []
    assert "无需复飞" in format_plan(plan)


def test_future_reference_date_creates_more_routine_work():
    near = plan_mission(REF)
    far = plan_mission((_d(REF) + timedelta(days=365)).isoformat())
    assert len(far.tasks) + len(far.deferred) >= len(near.tasks) + len(near.deferred)
