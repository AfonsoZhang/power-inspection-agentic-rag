"""策略图编译器：把自然语言作业意图编译成一张合法的策略图

这是 GaP 闭环里 LLM 唯一出现的位置，也是它相对"让 Agent 每轮自由决策"的关键差别：
模型只在**编译期**参与一次，产物是一张可校验的图；运行期由 policy_graph.execute
按图执行，不再有模型介入，因此可以离线、可复现、可审计。

编译不信任模型的输出，走 **生成 → 校验 → 带错重修** 的闭环：

    instruction ──▶ LLM ──▶ JSON 图 ──▶ validate()
                    ▲                      │
                    └──── 错误清单 ◀───────┘   最多重修 max_repairs 次
                                           │ 通过
                                           ▼
                                    MissionPolicy

校验器是确定性的（policy_graph.validate），所以"图合不合法"这件事从来不由模型说了算。
重修失败或没有 API Key 时回落到 default_policy()——降级成基线，而不是把非法的图放出去。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .policy_graph import MissionPolicy, PolicyError, default_policy
from .skills import describe_library

COMPILER_SYSTEM = """你是无人机巡检作业的**策略图编译器**。

把用户的作业意图编译成一张有向计算图（JSON），图的节点只能取自下方技能库。

{library}

## 硬约束（违反即编译失败并被打回重修）
1. 节点的 skill 必须是技能库里的名字；参数必须是该技能声明过的、且落在取值域内
2. 每个节点的全部 inputs 必须由它的前驱节点 outputs 提供（按端口类型名匹配）
3. 图必须无环，且恰好有一个产出 FlightOutcome 的终端节点
4. 任何 control 节点的上游必须存在 screen_airspace——未经合规筛查不得进入执行
5. order_route_nn 与 order_route_2opt 只能二选一

## 输出格式
只输出一个 JSON 对象，不要任何解释文字：
{{"name": "简短图名", "notes": "一句话说明这张图的取舍",
  "nodes": [{{"id": "p1", "skill": "sense_assets", "params": {{}}}}],
  "edges": [["p1", "t1"]]}}

## 编译提示
- 用户强调"稳"/"别断航"/"风大" → 调高 split_sorties.battery_reserve_pct 与 fly_sorties.rth_margin_min
- 用户强调"多飞几基"/"赶工期" → 调低余量、调高 max_towers_per_sortie，并在 notes 里写明这么做提高了中断风险
- 用户提到"顺路一起飞了" → 调高 collect_due_towers.lookahead_days
- 没有明确偏好就用技能默认参数"""


@dataclass
class CompileResult:
    policy: MissionPolicy
    attempts: int = 0
    errors_seen: list[list[str]] = field(default_factory=list)
    used_fallback: bool = False
    raw_responses: list[str] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        return self.attempts > 1 and not self.used_fallback


def compile_policy(
    instruction: str,
    *,
    complete_fn=None,
    max_repairs: int = 2,
) -> CompileResult:
    """把作业意图编译成策略图。

    complete_fn: (messages, system) -> 文本。默认走 llm_client.complete；
    注入自定义实现即可在无 API Key 的情况下测试整条重修闭环。
    """
    call = complete_fn or _default_complete
    system = COMPILER_SYSTEM.format(library=describe_library())
    messages = [{"role": "user", "content": f"作业意图：{instruction}"}]

    result = CompileResult(policy=default_policy())
    for attempt in range(1, max_repairs + 2):
        result.attempts = attempt
        try:
            text = call(messages, system)
        except Exception as e:  # noqa: BLE001 - 模型/网络不可用时降级到基线，不阻断作业
            result.errors_seen.append([f"调用模型失败：{type(e).__name__}: {e}"])
            result.used_fallback = True
            return result

        result.raw_responses.append(text)
        try:
            policy = MissionPolicy.from_dict(extract_json(text))
        except (ValueError, KeyError, TypeError) as e:
            errors = [f"输出不是可解析的策略图 JSON：{e}"]
        else:
            errors = policy.validate()
            if not errors:
                result.policy = policy
                return result

        result.errors_seen.append(errors)
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content":
                "上面这张图没有通过校验，问题如下，请修正后重新输出完整 JSON：\n- "
                + "\n- ".join(errors)},
        ]

    result.used_fallback = True
    return result


def extract_json(text: str) -> dict:
    """从模型输出里取出 JSON 对象：优先 ```json 围栏，其次第一个配平的大括号块。"""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        return json.loads(fence.group(1))

    start = text.find("{")
    if start < 0:
        raise ValueError("输出里找不到 JSON 对象")
    depth = 0
    in_str = escaped = False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("输出里的 JSON 对象没有闭合")


def _default_complete(messages: list[dict], system: str) -> str:
    from ..generation.llm_client import complete

    return complete(messages, system=system, temperature=0.0).text


def compile_and_check(instruction: str, **kwargs) -> MissionPolicy:
    """编译并要求成功——降级到基线时抛错，供不接受静默降级的调用方使用。"""
    result = compile_policy(instruction, **kwargs)
    if result.used_fallback:
        last = result.errors_seen[-1] if result.errors_seen else ["未知原因"]
        raise PolicyError("策略图编译失败，已用尽重修次数：\n- " + "\n- ".join(last))
    return result.policy
