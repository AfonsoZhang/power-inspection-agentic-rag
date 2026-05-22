"""规程 Markdown 加载与切片

切片策略：
- 优先按 Markdown 二级 / 三级标题切分（语义完整性优先）
- 单段超过 chunk_size 时再做滑窗式二次切分
- 每个 chunk 携带 (source, section_path) 元数据，用于引用溯源
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from ..config import load_config

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    section_path: str
    char_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return out


def load_regulation_chunks() -> list[Chunk]:
    cfg = load_config()
    reg_dir: Path = cfg["_paths"]["regulations_dir"]
    chunk_size = cfg["ingestion"]["chunk_size"]
    overlap = cfg["ingestion"]["chunk_overlap"]

    chunks: list[Chunk] = []
    for md_path in sorted(reg_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        sections = _parse_markdown_sections(text)
        for sec_path, sec_body in sections:
            for piece in _split_long(sec_body, chunk_size, overlap):
                cid = f"{md_path.stem}::{sec_path}::{len(chunks)}"
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        text=f"# {sec_path}\n{piece}".strip(),
                        source=md_path.name,
                        section_path=sec_path,
                        char_count=len(piece),
                    )
                )
    return chunks


def _parse_markdown_sections(text: str) -> list[tuple[str, str]]:
    """按标题层级把 Markdown 拆成 (section_path, body) 列表"""
    lines = text.splitlines()
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = []
    current_path: str | None = None
    current_body: list[str] = []

    def flush():
        if current_path is not None:
            sections.append((current_path, current_body[:]))

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_path = " > ".join(t for _, t in stack)
            current_body = []
        else:
            current_body.append(line)
    flush()

    return [(p, "\n".join(b).strip()) for p, b in sections if "".join(b).strip()]


def load_defect_cases() -> list[dict]:
    cfg = load_config()
    path: Path = cfg["_paths"]["defect_cases"]
    cases: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def case_to_text(case: dict) -> str:
    """把案例 JSON 拼接成可向量化的描述文本"""
    return (
        f"[案例 {case['case_id']}] "
        f"资产: {case.get('asset_id', '')} ({case.get('asset_type', '')}); "
        f"缺陷: {case.get('defect_type', '')}; "
        f"等级: {case.get('severity', '')}; "
        f"描述: {case.get('description', '')}; "
        f"处置: {case.get('treatment', '')}; "
        f"结果: {case.get('result', '')}; "
        f"参考规程: {case.get('regulation_ref', '')}; "
        f"标签: {', '.join(case.get('image_tags', []))}"
    )


def load_assets() -> dict[str, dict]:
    cfg = load_config()
    path: Path = cfg["_paths"]["asset_registry"]
    items = json.loads(path.read_text(encoding="utf-8"))
    return {a["asset_id"]: a for a in items}


def load_inspection_history() -> list[dict]:
    cfg = load_config()
    path: Path = cfg["_paths"]["inspection_history"]
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def get_asset_history(asset_id: str) -> list[dict]:
    return [h for h in load_inspection_history() if h["asset_id"] == asset_id]
