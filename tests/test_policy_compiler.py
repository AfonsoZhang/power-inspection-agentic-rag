"""策略图编译器：JSON 抽取与「生成 → 校验 → 带错重修」闭环

编译器唯一依赖模型的地方通过 complete_fn 注入，所以这里不需要 API Key 也能
把整条重修闭环跑穿——被测的正是"图合不合法由校验器说了算，不由模型说了算"。
"""
import json

import pytest

from src.policy.compiler import CompileResult, compile_and_check, compile_policy, extract_json
from src.policy.policy_graph import MissionPolicy, Node, PolicyError, default_policy

VALID = json.dumps(default_policy().to_dict(), ensure_ascii=False)
UNSAFE = json.dumps(MissionPolicy(
    "unsafe",
    [Node("p1", "sense_assets"), Node("t1", "collect_due_towers"),
     Node("r1", "order_route_2opt"), Node("s1", "split_sorties"), Node("f1", "fly_sorties")],
    [("p1", "t1"), ("t1", "r1"), ("r1", "s1"), ("s1", "f1")],
).to_dict(), ensure_ascii=False)


def _scripted(*replies):
    """把一串预设回复当成模型，并记录它每次收到的消息。"""
    seen = []

    def fn(messages, system):
        seen.append(messages)
        return replies[min(len(seen) - 1, len(replies) - 1)]

    fn.seen = seen
    return fn


# --------------------------------------------------------------------------
# JSON 抽取
# --------------------------------------------------------------------------

def test_extract_json_from_fenced_block():
    assert extract_json('说明\n```json\n{"a": 1}\n```\n结尾') == {"a": 1}


def test_extract_json_from_bare_prose():
    assert extract_json('这是图：{"a": {"b": 2}} 就这样') == {"a": {"b": 2}}


def test_extract_json_ignores_braces_inside_strings():
    assert extract_json('{"note": "先 { 再 }", "n": 1}')["n"] == 1


@pytest.mark.parametrize("text", ["完全没有 JSON", '{"未闭合": 1'])
def test_extract_json_rejects_unusable_output(text):
    with pytest.raises(ValueError):
        extract_json(text)


# --------------------------------------------------------------------------
# 重修闭环
# --------------------------------------------------------------------------

def test_valid_graph_compiles_on_the_first_try():
    r = compile_policy("正常排班", complete_fn=_scripted(VALID))
    assert r.attempts == 1 and not r.used_fallback and not r.repaired
    assert r.policy.validate() == []


def test_unsafe_graph_is_sent_back_with_the_error_list():
    fn = _scripted(UNSAFE, VALID)
    r = compile_policy("赶工期，多飞几基", complete_fn=fn)
    assert r.repaired and not r.used_fallback
    assert any("安全约束违规" in e for e in r.errors_seen[0])
    # 第二轮的对话里必须带上上一轮的图和错误清单，否则模型无从修起
    repair_turn = fn.seen[1]
    assert len(repair_turn) == 3
    assert "安全约束违规" in repair_turn[-1]["content"]


def test_unparseable_output_also_triggers_a_repair():
    r = compile_policy("随便", complete_fn=_scripted("我觉得不用图", VALID))
    assert r.repaired
    assert "不是可解析的策略图 JSON" in r.errors_seen[0][0]


def test_persistently_invalid_output_falls_back_to_the_baseline():
    r = compile_policy("随便", complete_fn=_scripted(UNSAFE), max_repairs=2)
    assert r.used_fallback and r.attempts == 3
    assert r.policy.signature() == default_policy().signature()


def test_model_failure_degrades_instead_of_raising():
    def boom(messages, system):
        raise TimeoutError("模型不可用")

    r = compile_policy("随便", complete_fn=boom)
    assert r.used_fallback and r.policy.validate() == []
    assert "TimeoutError" in r.errors_seen[0][0]


def test_compile_and_check_refuses_to_hide_a_fallback():
    with pytest.raises(PolicyError):
        compile_and_check("随便", complete_fn=_scripted(UNSAFE), max_repairs=0)


def test_compiled_policy_is_never_returned_unvalidated():
    """无论走哪条路径，交出去的图都必须是合法的。"""
    for fn in (_scripted(VALID), _scripted(UNSAFE, VALID), _scripted(UNSAFE), _scripted("胡说")):
        result: CompileResult = compile_policy("随便", complete_fn=fn, max_repairs=1)
        assert result.policy.validate() == []
