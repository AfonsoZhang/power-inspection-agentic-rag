"""工具定义完整性、错误处理，以及配置与磁盘的一致性"""
import json

import pytest

from src.agent.crew import ROLE_TOOLS
from src.agent.tools import TOOL_DEFINITIONS, TOOL_NAMES, execute_tool
from src.config import load_config, vlm_spec

# --------------------------------------------------------------------------
# 工具 schema
# --------------------------------------------------------------------------

def test_tool_names_are_unique():
    assert len(TOOL_NAMES) == len(set(TOOL_NAMES))


@pytest.mark.parametrize("tool", TOOL_DEFINITIONS, ids=lambda t: t["name"])
def test_every_tool_declares_a_usable_json_schema(tool):
    assert tool["description"].strip(), "工具缺少描述，模型无从判断何时调用"
    schema = tool["parameters"]
    assert schema["type"] == "object"
    for name in schema.get("required", []):
        assert name in schema["properties"], f"required 里的 {name} 未在 properties 中定义"
    for name, prop in schema["properties"].items():
        assert prop.get("type"), f"{name} 缺少 type"
        assert prop.get("description") or prop.get("items"), f"{name} 缺少描述"


def test_role_tool_scoping_only_references_real_tools():
    for role, names in ROLE_TOOLS.items():
        for n in names:
            assert n in TOOL_NAMES, f"{role} 引用了不存在的工具 {n}"


# --------------------------------------------------------------------------
# 执行与错误处理
# --------------------------------------------------------------------------

def test_unknown_tool_returns_readable_text_not_exception():
    assert "未知工具" in execute_tool("no_such_tool", {})


def test_tool_error_is_returned_as_text_for_the_model_to_recover():
    # 缺 required 参数会在 _dispatch 里抛 KeyError，必须被包成可读文本回灌
    out = execute_tool("search_regulations", {})
    assert "执行失败" in out and "KeyError" in out


def test_clearance_tool_without_assets_explains_itself():
    assert "未提供任何杆塔编号" in execute_tool("check_flight_clearance", {"asset_ids": []})


def test_deterministic_tools_run_without_any_api_key():
    """规划与合规两个工具是纯计算，没有 API Key 也必须能跑——CI 正是这么跑的。"""
    plan = execute_tool("plan_inspection_mission",
                        {"line_name": "济南西郊 110kV 输电线路", "reference_date": "2025-10-01"})
    assert "巡检任务计划" in plan
    clearance = execute_tool("check_flight_clearance", {"asset_ids": ["JN-110-052"], "agl_m": 120})
    assert "飞行前合规校验" in clearance


def test_lookup_asset_returns_json_for_known_and_message_for_unknown():
    known = execute_tool("lookup_asset", {"asset_id": "JN-110-052"})
    assert json.loads(known)["line_name"]
    assert "未找到资产" in execute_tool("lookup_asset", {"asset_id": "XX-000-000"})


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

def test_every_configured_path_exists_on_disk():
    paths = load_config()["_paths"]
    for key, path in paths.items():
        if key in ("project_root", "chroma_dir"):  # chroma_dir 由建索引时创建
            continue
        assert path.exists(), f"config.yaml 的 paths.{key} 指向不存在的位置: {path}"


def test_vlm_is_optional_and_absent_by_default():
    """默认配置不启用多模态，纯文本功能必须不受影响。"""
    assert vlm_spec() is None


def test_recheck_windows_are_strictly_ordered_by_severity():
    days = load_config()["mission"]["recheck_days_by_severity"]
    assert days["I"] < days["II"] < days["III"], "缺陷等级越高，复查时限应越短"


def test_battery_budget_can_hold_at_least_one_tower():
    m = load_config()["mission"]
    assert m["battery_minutes"] > m["hover_minutes_per_tower"] > 0
