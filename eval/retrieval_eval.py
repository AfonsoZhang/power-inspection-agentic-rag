"""检索质量评测：Recall@k / MRR / 命中率（**不需要任何 API Key**）

这是三个评测脚本里唯一不依赖大模型的一个——只用本地 sentence-transformers
和 ChromaDB，所以谁 clone 下来都能复现同一组数字。它衡量的是 RAG 的地基：
生成再好，检索没把该给的条款捞上来，答案就只能靠编。

参考答案 reference_chunks 有三种前缀，按来源分开评：
    "<规程文件名>::<章节路径前缀>"  → 规程库向量检索
    "case::<case_id>"               → 案例库向量检索
    "asset::<asset_id>"             → 资产档案的确定性查表

规程比对时，实际 chunk 的 section_path 前面还带一级 H1 文档标题，所以先剥掉 H1，
再判断参考路径是否为其前缀——同一节被二次切分成的多个 chunk 都算命中。

**评测范围只含纯文本题**：golden_qa 里带 image_path 的 5 题输入是图像，
用题干做文本检索衡量不到它们的真实链路，计入只会虚高或虚低，所以单独剔除并在结果里标明。
asset:: 参考走的是确定性查表（命中与否只取决于档案库有没有这条记录），
不属于"检索质量"，同样单列不混入 Recall。

用法：python eval/retrieval_eval.py [--k 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN_PATH = ROOT / "eval" / "golden_qa.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"


def load_samples() -> list[dict]:
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strip_doc_title(section_path: str) -> str:
    """剥掉 chunk section_path 最前面那一级 H1 文档标题。"""
    head, sep, rest = section_path.partition(" > ")
    return rest if sep else head


def chunk_matches(chunk_id: str, reference: str) -> bool:
    ref_source, _, ref_section = reference.partition("::")
    source, _, tail = chunk_id.partition("::")
    if source != ref_source:
        return False
    section = strip_doc_title(tail.rsplit("::", 1)[0])
    return section == ref_section or section.startswith(ref_section + " > ")


def evaluate(k: int) -> dict:
    # 检索栈是重依赖，放在函数内 import，好让 chunk_matches 这类纯函数能被单测直接引用
    from src.retrieval.retriever import retrieve_asset_card, retrieve_cases, retrieve_regulations

    samples = load_samples()
    # 带图的题目输入是图像，用题干做文本检索衡量不到真实链路，单独剔除
    graded = [s for s in samples if s.get("reference_chunks") and not s.get("image_path")]
    skipped = [s["qid"] for s in samples if s.get("image_path")]

    per_sample = []
    for s in graded:
        refs = s["reference_chunks"]
        reg_refs = [r for r in refs if not r.startswith(("case::", "asset::"))]
        case_refs = [r.split("::", 1)[1] for r in refs if r.startswith("case::")]
        asset_refs = [r.split("::", 1)[1] for r in refs if r.startswith("asset::")]

        ranked: list[str] = []   # 归一化后的召回结果标识，按排名
        if reg_refs:
            ranked += [h["id"] for h in retrieve_regulations(s["question"], top_k=k)]
        if case_refs:
            ranked += [f"case::{h['id']}" for h in retrieve_cases(s["question"], top_k=k)]

        matched: set[str] = set()
        first_rank = None
        for rank, item in enumerate(ranked, 1):
            for ref in reg_refs:
                if chunk_matches(item, ref):
                    matched.add(ref)
                    first_rank = first_rank or rank
            for cid in case_refs:
                if item == f"case::{cid}":
                    matched.add(f"case::{cid}")
                    first_rank = first_rank or rank

        n_refs = len(reg_refs) + len(case_refs)
        per_sample.append({
            "qid": s["qid"],
            "n_refs": n_refs,
            "n_matched": len(matched),
            "recall": len(matched) / n_refs if n_refs else None,
            "first_hit_rank": first_rank,
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
            "missed": sorted((set(reg_refs) | {f"case::{c}" for c in case_refs}) - matched),
            "asset_lookups": {a: retrieve_asset_card(a) is not None for a in asset_refs},
        })

    scored = [r for r in per_sample if r["recall"] is not None]
    n = len(scored)
    asset_checks = [ok for r in per_sample for ok in r["asset_lookups"].values()]
    summary = {
        "k": k,
        "n_questions": n,
        "skipped_multimodal": skipped,
        "recall_at_k": round(sum(r["recall"] for r in scored) / n, 4),
        "mrr": round(sum(r["reciprocal_rank"] for r in scored) / n, 4),
        "hit_rate": round(sum(1 for r in scored if r["first_hit_rank"]) / n, 4),
        "full_recall_rate": round(sum(1 for r in scored if r["recall"] == 1.0) / n, 4),
        "asset_lookup_success": (round(sum(asset_checks) / len(asset_checks), 4)
                                 if asset_checks else None),
    }
    return {"summary": summary, "per_sample": per_sample}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=6, help="召回条数")
    args = parser.parse_args()

    result = evaluate(args.k)
    s = result["summary"]
    print(f"检索评测（k={s['k']}，{s['n_questions']} 题）")
    print(f"  Recall@{s['k']} : {s['recall_at_k']:.3f}")
    print(f"  MRR        : {s['mrr']:.3f}")
    print(f"  命中率      : {s['hit_rate']:.3f}（至少命中一条参考章节）")
    print(f"  全召回率    : {s['full_recall_rate']:.3f}（参考章节全部召回）")
    if s["asset_lookup_success"] is not None:
        print(f"  资产查表    : {s['asset_lookup_success']:.3f}（确定性查表，不计入上面的检索指标）")
    if s["skipped_multimodal"]:
        print(f"  已剔除多模态题：{', '.join(s['skipped_multimodal'])}（输入是图像，非文本检索链路）")

    missed = [r for r in result["per_sample"] if r["recall"] is not None and r["recall"] < 1.0]
    if missed:
        print("\n未完全召回的题目：")
        for r in missed:
            print(f"  {r['qid']}  {r['n_matched']}/{r['n_refs']}  漏: {r['missed']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval_k{args.k}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
