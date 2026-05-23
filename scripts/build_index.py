"""一键构建向量索引

执行：
    python scripts/build_index.py

会重置以下两个 collection 并重新写入：
- regulations
- defect_cases
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tqdm import tqdm  # noqa: E402

from src.config import load_config  # noqa: E402
from src.generation.llm_client import embed_texts  # noqa: E402
from src.ingestion.text_loader import (  # noqa: E402
    case_to_text,
    load_defect_cases,
    load_regulation_chunks,
)
from src.retrieval.vector_store import reset_collection, upsert  # noqa: E402


def build_regulations_index() -> int:
    cfg = load_config()
    coll = reset_collection(cfg["vector_store"]["regulations_collection"])
    chunks = load_regulation_chunks()
    if not chunks:
        print("[!] 未找到任何规程文件")
        return 0

    texts = [c.text for c in chunks]
    print(f"[规程] 共 {len(texts)} 个 chunk，开始向量化...")
    embeddings = embed_texts_with_progress(texts)

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {"source": c.source, "section_path": c.section_path, "char_count": c.char_count}
        for c in chunks
    ]
    upsert(coll, ids, embeddings, documents, metadatas)
    print(f"[规程] 写入完成: {len(ids)} 条")
    return len(ids)


def build_cases_index() -> int:
    cfg = load_config()
    coll = reset_collection(cfg["vector_store"]["cases_collection"])
    cases = load_defect_cases()
    if not cases:
        print("[!] 未找到任何案例")
        return 0

    documents = [case_to_text(c) for c in cases]
    print(f"[案例] 共 {len(documents)} 条，开始向量化...")
    embeddings = embed_texts_with_progress(documents)

    ids = [c["case_id"] for c in cases]
    metadatas = [
        {
            "case_id": c["case_id"],
            "asset_id": c.get("asset_id"),
            "asset_type": c.get("asset_type"),
            "defect_type": c.get("defect_type"),
            "severity": c.get("severity"),
            "regulation_ref": c.get("regulation_ref"),
            "treatment_days": c.get("treatment_days"),
            "tags": c.get("image_tags", []),
        }
        for c in cases
    ]
    upsert(coll, ids, embeddings, documents, metadatas)
    print(f"[案例] 写入完成: {len(ids)} 条")
    return len(ids)


def embed_texts_with_progress(texts: list[str]) -> list[list[float]]:
    batch_size = 32
    out: list[list[float]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embedding"):
        batch = texts[i : i + batch_size]
        out.extend(embed_texts(batch))
    return out


def main() -> None:
    t0 = time.time()
    n_reg = build_regulations_index()
    n_case = build_cases_index()
    print(f"\n构建完成: 规程 {n_reg} chunks, 案例 {n_case} 条, 总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
