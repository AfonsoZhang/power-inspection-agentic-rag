"""业务 KPI 测算脚本

不依赖 RAGAS，仅做：
- 平均问答耗时
- 引用命中率（answer 中是否出现 reference 关键词或 case_id）
- 等级判定准确率（如果 ground_truth 含等级关键字）
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generation.report_generator import answer_question  # noqa: E402

GOLDEN_PATH = ROOT / "eval" / "golden_qa.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"

LEVEL_RE = re.compile(r"(I{1,3}|1|2|3)\s*级")
CASE_RE = re.compile(r"C\d{4}")
SECTION_RE = re.compile(r"§\s*\d+(?:\.\d+)*")


def load_samples() -> list[dict]:
    out: list[dict] = []
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def normalize_level(text: str) -> set[str]:
    levels = set()
    for m in LEVEL_RE.findall(text):
        levels.add({"1": "I", "2": "II", "3": "III"}.get(m, m))
    return levels


def evaluate_one(sample: dict) -> dict:
    t0 = time.time()
    try:
        resp = answer_question(sample["question"])
    except Exception as e:
        return {"qid": sample["qid"], "error": str(e), "elapsed_s": -1}

    elapsed = round(time.time() - t0, 2)
    answer = resp["answer"]
    contexts = resp["contexts"]

    gt_levels = normalize_level(sample["ground_truth"])
    ans_levels = normalize_level(answer)
    level_match = bool(gt_levels) and gt_levels.issubset(ans_levels)

    expected_cases = set(CASE_RE.findall(sample["ground_truth"]))
    actual_cases = set(CASE_RE.findall(answer))
    case_recall = (len(expected_cases & actual_cases) / len(expected_cases)) if expected_cases else None

    has_citation = bool(SECTION_RE.search(answer)) or bool(actual_cases)

    return {
        "qid": sample["qid"],
        "question": sample["question"],
        "elapsed_s": elapsed,
        "level_match": level_match if gt_levels else None,
        "case_recall": case_recall,
        "has_citation": has_citation,
        "n_contexts": len(contexts),
        "answer_preview": answer[:200],
    }


def main() -> None:
    samples = load_samples()
    print(f"载入 Golden 样本 {len(samples)} 条\n")
    results = []
    for s in samples:
        r = evaluate_one(s)
        results.append(r)
        flags = []
        if r.get("level_match") is True:
            flags.append("等级√")
        elif r.get("level_match") is False:
            flags.append("等级×")
        if r.get("case_recall") is not None:
            flags.append(f"案例召回 {r['case_recall']:.0%}")
        if r.get("has_citation"):
            flags.append("有引用")
        print(f"[{r['qid']}] {r['elapsed_s']}s {' / '.join(flags)}")

    valid = [r for r in results if r.get("elapsed_s", -1) >= 0]
    avg_latency = sum(r["elapsed_s"] for r in valid) / max(len(valid), 1)
    citation_rate = sum(1 for r in valid if r.get("has_citation")) / max(len(valid), 1)
    level_items = [r for r in valid if r.get("level_match") is not None]
    level_acc = sum(1 for r in level_items if r["level_match"]) / max(len(level_items), 1)
    case_items = [r for r in valid if r.get("case_recall") is not None]
    case_recall_avg = sum(r["case_recall"] for r in case_items) / max(len(case_items), 1)

    summary = {
        "total": len(samples),
        "valid": len(valid),
        "avg_latency_s": round(avg_latency, 2),
        "citation_rate": round(citation_rate, 3),
        "level_accuracy": round(level_acc, 3),
        "case_recall_avg": round(case_recall_avg, 3),
    }
    print("\n业务 KPI 汇总:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"business_kpi_{int(time.time())}.json"
    out.write_text(
        json.dumps({"summary": summary, "details": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
