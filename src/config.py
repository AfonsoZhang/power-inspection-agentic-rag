"""加载 config.yaml 与 .env，提供全局配置访问入口"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"未找到配置文件: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["_env"] = {
        "api_key": os.getenv("MIMO_API_KEY", "").strip(),
        "llm_base_url": os.getenv("LLM_BASE_URL", cfg["provider"]["base_url"]).strip(),
        "debug": os.getenv("DEBUG", "0") == "1",
    }
    cfg["_paths"] = {
        "project_root": PROJECT_ROOT,
        "regulations_dir": PROJECT_ROOT / cfg["paths"]["regulations_dir"].lstrip("./"),
        "defect_cases": PROJECT_ROOT / cfg["paths"]["defect_cases"].lstrip("./"),
        "asset_registry": PROJECT_ROOT / cfg["paths"]["asset_registry"].lstrip("./"),
        "inspection_history": PROJECT_ROOT / cfg["paths"]["inspection_history"].lstrip("./"),
        "chroma_dir": PROJECT_ROOT / cfg["vector_store"]["persist_dir"].lstrip("./"),
    }
    return cfg


def get_api_key() -> str:
    cfg = load_config()
    key = cfg["_env"]["api_key"]
    if not key or key.startswith("sk-your"):
        raise RuntimeError(
            "未检测到有效的 MIMO_API_KEY。\n"
            "请在 .env 文件中填入你的 MiMo API Key。"
        )
    return key
