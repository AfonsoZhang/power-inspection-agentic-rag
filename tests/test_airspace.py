"""飞行前合规校验：空域限制、气象限值、总裁决取最严"""
from src.ingestion.text_loader import load_assets
from src.mission.airspace import (
    check_asset,
    check_flight,
    check_weather,
    format_report,
    load_constraints,
)

GOOD_WEATHER = {"condition": "晴", "wind_mps": 5.0, "gust_mps": 7.0,
                "visibility_km": 10.0, "temperature_c": 15.0}


# --------------------------------------------------------------------------
# 气象
# --------------------------------------------------------------------------

def test_good_weather_passes():
    assert check_weather(GOOD_WEATHER).verdict == "allowed"


def test_missing_weather_is_unknown_not_allowed():
    v = check_weather(None)
    assert v.verdict == "unknown" and v.reasons


def test_prohibited_condition_blocks_flight():
    v = check_weather({**GOOD_WEATHER, "condition": "雷暴转阵雨"})
    assert v.verdict == "forbidden"
    assert any("雷暴" in r for r in v.reasons)


def test_each_numeric_limit_is_enforced_independently():
    limits = load_constraints()["weather_limits"]
    cases = [
        ({"wind_mps": limits["max_wind_mps"] + 1}, "风速"),
        ({"gust_mps": limits["max_gust_mps"] + 1}, "阵风"),
        ({"visibility_km": limits["min_visibility_km"] - 1}, "能见度"),
        ({"temperature_c": limits["min_temperature_c"] - 1}, "气温"),
    ]
    for override, keyword in cases:
        v = check_weather({**GOOD_WEATHER, **override})
        assert v.verdict == "forbidden", f"{keyword} 超限未被拦截"
        assert any(keyword in r for r in v.reasons)


def test_values_exactly_at_limit_are_allowed():
    limits = load_constraints()["weather_limits"]
    v = check_weather({**GOOD_WEATHER,
                       "wind_mps": limits["max_wind_mps"],
                       "visibility_km": limits["min_visibility_km"]})
    assert v.verdict == "allowed"


# --------------------------------------------------------------------------
# 空域
# --------------------------------------------------------------------------

def _asset_in_zone(zone_id):
    for aid in load_assets():
        c = check_asset(aid, 120.0)
        if any(z.zone_id == zone_id for z in c.zones):
            return c
    return None


def test_no_fly_zone_forbids_regardless_of_requested_height():
    c = _asset_in_zone("RST-QD-021")
    assert c is not None, "演示数据里应有塔位落在禁飞区内"
    assert c.verdict == "forbidden" and c.allowed_agl_m == 0.0


def test_height_limited_zone_downgrades_to_restricted():
    c = _asset_in_zone("RST-JN-010")
    assert c is not None
    assert c.verdict == "restricted"
    assert c.allowed_agl_m == 60.0 < c.requested_agl_m


def test_low_enough_request_inside_limited_zone_is_allowed():
    c = _asset_in_zone("RST-JN-010")
    relaxed = check_asset(c.asset_id, 50.0)
    assert relaxed.verdict == "allowed" and relaxed.allowed_agl_m == 50.0


def test_unknown_asset_is_forbidden_not_silently_passed():
    c = check_asset("XX-999-999", 120.0)
    assert c.verdict == "forbidden"
    assert "不在档案库" in " ".join(c.reasons)


def test_emi_distance_follows_voltage_level():
    table = load_constraints()["emi_safe_distance_m"]
    assets = load_assets()
    for aid, a in assets.items():
        c = check_asset(aid, 120.0)
        assert c.emi_min_distance_m == table[str(a["voltage_kv"])]


# --------------------------------------------------------------------------
# 总裁决
# --------------------------------------------------------------------------

def test_overall_verdict_takes_the_strictest():
    ids = list(load_assets())
    report = check_flight(ids, agl_m=120.0, weather=GOOD_WEATHER)
    assert report.verdict == "forbidden"  # 数据里存在禁飞塔位
    assert any(a.verdict == "forbidden" for a in report.assets)


def test_bad_weather_alone_blocks_even_clear_airspace():
    clear = [a for a in load_assets() if check_asset(a, 120.0).verdict == "allowed"][:2]
    report = check_flight(clear, agl_m=120.0, weather={**GOOD_WEATHER, "condition": "冰雹"})
    assert report.verdict == "forbidden"


def test_report_lists_blocked_assets_in_recommendations():
    ids = list(load_assets())
    report = check_flight(ids, agl_m=120.0, weather=GOOD_WEATHER)
    blocked = [a.asset_id for a in report.assets if a.verdict == "forbidden"]
    joined = " ".join(report.recommendations)
    for aid in blocked:
        assert aid in joined


def test_format_report_carries_the_synthetic_data_disclaimer():
    text = format_report(check_flight(list(load_assets())[:3], weather=GOOD_WEATHER))
    assert "合成演示数据" in text
