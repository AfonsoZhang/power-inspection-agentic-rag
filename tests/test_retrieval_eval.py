"""检索评测的匹配判据（纯函数，不触碰向量库）"""
import json

from eval.retrieval_eval import GOLDEN_PATH, chunk_matches, load_samples, strip_doc_title

CHUNK = "导线及金具缺陷判定标准::架空输电线路导线及金具缺陷判定与处置标准（合成） > 第一章 导线缺陷 > 1.1 导线断股 > 1.1.1 判定标准::1"


def test_strip_doc_title_removes_only_the_h1_level():
    assert strip_doc_title("文档标题 > 第一章 > 1.1 节") == "第一章 > 1.1 节"
    assert strip_doc_title("只有一级") == "只有一级"


def test_reference_matches_a_deeper_chunk_of_the_same_section():
    assert chunk_matches(CHUNK, "导线及金具缺陷判定标准::第一章 导线缺陷 > 1.1 导线断股")


def test_reference_does_not_match_a_sibling_section():
    assert not chunk_matches(CHUNK, "导线及金具缺陷判定标准::第一章 导线缺陷 > 1.2 导线散股")


def test_reference_does_not_match_across_documents():
    assert not chunk_matches(CHUNK, "杆塔本体巡检规范::第一章 导线缺陷 > 1.1 导线断股")


def test_prefix_match_requires_a_section_boundary():
    """'1.1 导线断' 不是 '1.1 导线断股' 的合法章节前缀，不能算命中。"""
    assert not chunk_matches(CHUNK, "导线及金具缺陷判定标准::第一章 导线缺陷 > 1.1 导线断")


def test_golden_set_is_well_formed():
    samples = load_samples()
    assert len(samples) == sum(1 for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if line.strip())
    qids = [s["qid"] for s in samples]
    assert len(qids) == len(set(qids)), "golden QA 存在重复 qid"
    for s in samples:
        assert s["question"].strip() and s["ground_truth"].strip()
        assert s["reference_chunks"], f"{s['qid']} 没有参考章节，无法评分"


def test_referenced_case_ids_exist_in_the_case_library():
    """参考答案引用的 case_id 必须真实存在，否则评测在跟不存在的东西比。"""
    case_ids = {json.loads(line)["case_id"]
                for line in (GOLDEN_PATH.parent.parent / "data/defects_history/cases.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()}
    for s in load_samples():
        for ref in s["reference_chunks"]:
            if ref.startswith("case::"):
                assert ref.split("::", 1)[1] in case_ids, f"{s['qid']} 引用了不存在的案例 {ref}"
