"""RAG 质量评测（LLM-as-Judge，三模式对比）

对 golden_qa.jsonl 中的每个问题，分三种模式运行：
1. Basic RAG（纯文本，固定检索→生成）
2. Agent RAG（纯文本，LLM 自主调用工具）
3. Agent RAG + 图像（多模态，仅图像题）

评委：mimo-v2.5-pro（纯文本打分），三个维度：
- Faithfulness: 答案是否忠实于检索上下文，无编造
- Answer Relevancy: 答案是否切题、完整
- Context Precision: 检索到的上下文是否相关

用法：python eval/ragas_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.agent import run_agent  # noqa: E402
from src.generation.llm_client import chat  # noqa: E402
from src.generation.prompts import FAITHFULNESS_RUBRIC  # noqa: E402
from src.generation.report_generator import answer_question  # noqa: E402

GOLDEN_PATH = ROOT / "eval" / "golden_qa.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"

# 忠实度判据复用 src/generation/prompts.py 的共享 rubric，与在线 grade 节点同尺
FAITHFULNESS_PROMPT = """你是一个严格的 RAG 系统评测专家。请判断以下【回答】是否忠实于【检索上下文】。

""" + FAITHFULNESS_RUBRIC + """

【问题】
{question}

【检索上下文】
{contexts}

【回答】
{answer}

请只输出一个 JSON 对象，格式：{{"score": <1-5>, "reason": "<一句话理由>"}}"""

RELEVANCY_PROMPT = """你是一个严格的 RAG 系统评测专家。请判断以下【回答】是否切题且完整地回答了【问题】。

评分标准（1-5 分）：
5 = 完整、准确地回答了问题，信息充分
4 = 基本回答了问题，略有遗漏
3 = 部分回答了问题，有明显遗漏
2 = 回答偏题或严重不完整
1 = 完全没有回答问题

【问题】
{question}

【标准答案（参考）】
{ground_truth}

【系统回答】
{answer}

请只输出一个 JSON 对象，格式：{{"score": <1-5>, "reason": "<一句话理由>"}}"""

CONTEXT_PRECISION_PROMPT = """你是一个严格的 RAG 系统评测专家。请判断以下【检索上下文】对于回答【问题】是否相关、精准。

评分标准（1-5 分）：
5 = 所有检索结果都高度相关，无噪声
4 = 大部分检索结果相关，少量噪声
3 = 约一半检索结果相关
2 = 大部分检索结果不相关
1 = 检索结果与问题无关

【问题】
{question}

【检索上下文】
{contexts}

请只输出一个 JSON 对象，格式：{{"score": <1-5>, "reason": "<一句话理由>"}}"""


def load_golden() -> list[dict]:
    items: list[dict] = []
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_basic_rag(question: str) -> dict:
    t0 = time.time()
    resp = answer_question(question)
    return {
        "answer": resp["answer"],
        "contexts": [c.get("document", "")[:500] for c in resp["contexts"]],
        "elapsed_s": round(time.time() - t0, 2),
    }


def run_agent_text(question: str) -> dict:
    t0 = time.time()
    result = run_agent(question)
    contexts = []
    for step in result.steps:
        if step.step_type == "tool_result":
            contexts.append(step.content)
    return {
        "answer": result.answer,
        "contexts": contexts,
        "turns": result.total_turns,
        "elapsed_s": round(time.time() - t0, 2),
    }


def run_agent_image(question: str, image_path: str) -> dict:
    t0 = time.time()
    full_path = str(ROOT / image_path)
    result = run_agent(question, image_path=full_path)
    contexts = []
    for step in result.steps:
        if step.step_type == "tool_result":
            contexts.append(step.content)
    return {
        "answer": result.answer,
        "contexts": contexts,
        "turns": result.total_turns,
        "elapsed_s": round(time.time() - t0, 2),
    }


def _parse_judge_response(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"score": 0, "reason": "解析失败"}


def judge_one(question: str, answer: str, contexts: list[str],
              ground_truth: str) -> dict:
    ctx_text = "\n\n".join(
        f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts)
    ) or "（无检索上下文）"

    scores = {}
    for metric, prompt_tpl in [
        ("faithfulness", FAITHFULNESS_PROMPT),
        ("answer_relevancy", RELEVANCY_PROMPT),
        ("context_precision", CONTEXT_PRECISION_PROMPT),
    ]:
        prompt = prompt_tpl.format(
            question=question,
            answer=answer or "（无回答）",
            contexts=ctx_text,
            ground_truth=ground_truth,
        )
        try:
            resp = chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            parsed = _parse_judge_response(resp)
            scores[metric] = parsed.get("score", 0)
            scores[f"{metric}_reason"] = parsed.get("reason", "")
        except Exception as e:
            scores[metric] = 0
            scores[f"{metric}_reason"] = str(e)

    return scores


def collect_and_judge(samples: list[dict]) -> dict:
    results = {"basic_rag": [], "agent_text": [], "agent_image": []}

    text_samples = [s for s in samples if "image_path" not in s]
    image_samples = [s for s in samples if "image_path" in s]

    print(f"\n文本题 {len(text_samples)} 条，图像题 {len(image_samples)} 条\n")

    # --- 文本题：Basic RAG + Agent ---
    for s in text_samples:
        qid = s["qid"]

        # Basic RAG
        print(f"[{qid}] Basic RAG...")
        try:
            br = run_basic_rag(s["question"])
            scores = judge_one(s["question"], br["answer"], br["contexts"], s["ground_truth"])
            results["basic_rag"].append({
                "qid": qid, "elapsed_s": br["elapsed_s"], **scores,
                "answer_preview": br["answer"][:200],
            })
            print(f"  -> {br['elapsed_s']}s | F={scores['faithfulness']} R={scores['answer_relevancy']} C={scores['context_precision']}")
        except Exception as e:
            print(f"  -> FAIL: {e}")
            results["basic_rag"].append({"qid": qid, "error": str(e)})

        # Agent (text)
        print(f"[{qid}] Agent RAG...")
        try:
            ar = run_agent_text(s["question"])
            scores = judge_one(s["question"], ar["answer"], ar["contexts"], s["ground_truth"])
            results["agent_text"].append({
                "qid": qid, "elapsed_s": ar["elapsed_s"], "turns": ar["turns"], **scores,
                "answer_preview": ar["answer"][:200],
            })
            print(f"  -> {ar['elapsed_s']}s {ar['turns']}轮 | F={scores['faithfulness']} R={scores['answer_relevancy']} C={scores['context_precision']}")
        except Exception as e:
            print(f"  -> FAIL: {e}")
            results["agent_text"].append({"qid": qid, "error": str(e)})

    # --- 图像题：Agent + Image ---
    for s in image_samples:
        qid = s["qid"]
        print(f"[{qid}] Agent + 图像 ({s['image_path']})...")
        try:
            ir = run_agent_image(s["question"], s["image_path"])
            scores = judge_one(s["question"], ir["answer"], ir["contexts"], s["ground_truth"])
            results["agent_image"].append({
                "qid": qid, "image": s["image_path"],
                "elapsed_s": ir["elapsed_s"], "turns": ir["turns"], **scores,
                "answer_preview": ir["answer"][:200],
            })
            print(f"  -> {ir['elapsed_s']}s {ir['turns']}轮 | F={scores['faithfulness']} R={scores['answer_relevancy']} C={scores['context_precision']}")
        except Exception as e:
            print(f"  -> FAIL: {e}")
            results["agent_image"].append({"qid": qid, "error": str(e)})

    return results


def compute_summary(items: list[dict]) -> dict:
    valid = [d for d in items if "error" not in d and d.get("faithfulness", 0) > 0]
    if not valid:
        return {"n": 0}
    n = len(valid)
    avg = lambda key: round(sum(d.get(key, 0) for d in valid) / n, 2)
    return {
        "n": n,
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "avg_latency_s": avg("elapsed_s"),
    }


def main() -> None:
    samples = load_golden()
    print(f"载入 Golden 样本 {len(samples)} 条")

    results = collect_and_judge(samples)

    summaries = {}
    print("\n" + "=" * 60)
    print("评测结果汇总（1-5 分制）")
    print("=" * 60)

    for mode, label in [
        ("basic_rag", "基础 RAG（纯文本）"),
        ("agent_text", "Agent RAG（纯文本）"),
        ("agent_image", "Agent RAG + 图像"),
    ]:
        s = compute_summary(results[mode])
        summaries[mode] = s
        if s["n"] == 0:
            print(f"\n{label}: 无有效结果")
            continue
        print(f"\n{label} ({s['n']} 题):")
        print(f"  Faithfulness:      {s['faithfulness']:.2f} / 5  ({s['faithfulness']/5:.0%})")
        print(f"  Answer Relevancy:  {s['answer_relevancy']:.2f} / 5  ({s['answer_relevancy']/5:.0%})")
        print(f"  Context Precision: {s['context_precision']:.2f} / 5  ({s['context_precision']/5:.0%})")
        print(f"  平均耗时:          {s['avg_latency_s']:.1f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"ragas_{int(time.time())}.json"
    out.write_text(
        json.dumps(
            {"summaries": summaries, "details": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n详细结果已写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
