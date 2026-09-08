"""策略图自学习评测：在仿真里搜出比基线更抗扰动的作业图（**不需要任何 API Key**）

与 retrieval_eval.py 一样，这个脚本不调大模型：搜索空间由技能库取值域界定，
评价由内置仿真给出，全程确定性，谁 clone 下来都能复现同一组数字。

产物两份：
    eval/results/policy_search.json      基线 vs 最优的完整对照，含被安全判据否决的候选
    data/policies/learned_policy.json    最优图本身——可被 policy_graph.active_policy() 直接加载执行

注意仿真口径：随机模型系数是工程假设值而非实测标定（见 src/policy/simulator.py 顶部），
所以这里的绝对数字只在本仓库口径下有意义，用于说明「图可被搜索、验证与固化」这一机制。

用法：python eval/policy_search.py [--trials 30] [--line "济南西郊 110kV 输电线路"]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.policy.selflearn import format_learning, search  # noqa: E402

# 评测用的标准气象条件：晴、5 m/s 风。写死是为了让结果可比——
# 换了天气就是换了题，不能拿两次结果直接比大小。
BASELINE_WEATHER = {
    "condition": "晴",
    "wind_mps": 5.0,
    "gust_mps": 7.0,
    "visibility_km": 10.0,
    "temperature_c": 15.0,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2025-10-01", help="计划参考日期")
    ap.add_argument("--line", default=None, help="限定线路，缺省为全部线路")
    ap.add_argument("--trials", type=int, default=30, help="单张图的蒙特卡洛次数")
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    report = search(args.date, args.line, BASELINE_WEATHER, trials=args.trials, seed=args.seed)
    print(format_learning(report))

    out = ROOT / "eval" / "results" / "policy_search.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    learned = ROOT / "data" / "policies" / "learned_policy.json"
    learned.parent.mkdir(parents=True, exist_ok=True)
    policy = report.best_policy.to_dict()
    policy["notes"] = (
        f"由 eval/policy_search.py 在仿真中搜出（{report.evaluated} 张候选图 × "
        f"{report.trials} 次蒙特卡洛，seed={args.seed}，标准气象 5 m/s）。"
        f"相对基线：coverage {report.baseline.coverage:.1%} → {report.best.coverage:.1%}，"
        f"备降率 {report.baseline.incident_rate:.1%} → {report.best.incident_rate:.1%}。"
    )
    learned.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n结果已写入 {out.relative_to(ROOT)}")
    print(f"最优图已固化到 {learned.relative_to(ROOT)}（active_policy() 会自动加载）")


if __name__ == "__main__":
    main()
