"""意图路由：规则分流与实体抽取"""
import pytest

from src.router.intent_router import (
    detect_intent,
    extract_asset_id,
    extract_asset_ids,
    extract_line_name,
)


@pytest.mark.parametrize("question,expected", [
    ("JN-110-052 之前巡检发现过什么问题？", "ask_history"),
    ("QD-110-103 的档案给我看下", "ask_history"),
    ("济南西郊 110kV 输电线路明天怎么排架次？", "plan_mission"),
    ("哪些杆塔需要复飞？", "plan_mission"),
    ("JN-220-018 今天能不能飞？", "flight_clearance"),
    ("这条线路有没有禁飞区限制", "flight_clearance"),
    ("复合绝缘子伞裙撕裂属于几级缺陷？", "ask_regulation"),
    ("无人机巡检一般用什么镜头", "ask_general"),
])
def test_detect_intent(question, expected):
    assert detect_intent(question) == expected


def test_clearance_wins_over_plan_when_both_keywords_present():
    """能不能飞是硬约束，同时命中时应先走合规校验。"""
    assert detect_intent("排架次之前先看看空域能不能飞") == "flight_clearance"


def test_extract_asset_id_returns_first_match():
    assert extract_asset_id("对比 JN-110-052 和 QD-110-103") == "JN-110-052"
    assert extract_asset_id("没有编号") is None


def test_extract_asset_ids_dedupes_and_preserves_order():
    q = "校验 QD-110-103、JN-110-052、QD-110-103 这几基"
    assert extract_asset_ids(q) == ["QD-110-103", "JN-110-052"]


def test_extract_line_name():
    assert extract_line_name("济南西郊 110kV 输电线路明天飞几架次") == "济南西郊 110kV 输电线路"
    assert extract_line_name("随便问问") is None
