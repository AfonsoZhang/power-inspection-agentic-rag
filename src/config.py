"""加载 config.yaml 与 .env，提供全局配置访问入口

对外主要有三个入口：
- ``load_config()``  : 解析后的配置字典（含 ``_paths`` 绝对路径）
- ``llm_spec()``     : 文本推理模型的 ProviderSpec
- ``vlm_spec()``     : 多模态模型的 ProviderSpec（未配置时返回 None）
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .generation.providers import DEFAULT_BASE_URLS, ProviderSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

# 通用回落 key，未按 provider 单独配置时使用
GENERIC_KEY_ENVS = ("LLM_API_KEY",)

_PATH_KEYS = (
    "regulations_dir",
    "defect_cases",
    "asset_registry",
    "inspection_history",
    "airspace_constraints",
    "sample_images",
)


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"未找到配置文件: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "provider" in cfg and "llm" not in cfg:
        raise RuntimeError(
            "检测到 v1 的 `provider:` 配置段。v2 已拆成 `llm:` / `vlm:` / `embedding:` 三段，"
            "请参照仓库内最新的 config.yaml 更新。"
        )

    cfg["_env"] = {"debug": os.getenv("DEBUG", "0") == "1"}
    cfg["_paths"] = {
        "project_root": PROJECT_ROOT,
        "chroma_dir": _abs(cfg["vector_store"]["persist_dir"]),
        **{k: _abs(cfg["paths"][k]) for k in _PATH_KEYS},
    }
    return cfg


def _abs(rel: str) -> Path:
    return PROJECT_ROOT / rel.lstrip("./")


def _resolve_key(section: dict) -> str:
    """按 api_key_env -> 通用 env 的顺序找 key。"""
    names = [section.get("api_key_env") or "", *GENERIC_KEY_ENVS]
    for name in names:
        if not name:
            continue
        value = os.getenv(name, "").strip()
        if value and not value.startswith("sk-your"):
            return value
    return ""


def _spec(section: dict) -> ProviderSpec:
    provider = section["provider"]
    base_url = (section.get("base_url") or "").strip() or DEFAULT_BASE_URLS.get(provider, "")
    return ProviderSpec(
        provider=provider,
        model=section["model"],
        base_url=base_url,
        api_key=_resolve_key(section),
        timeout=section.get("request_timeout", 120),
        max_retries=section.get("max_retries", 3),
    )


def llm_spec() -> ProviderSpec:
    cfg = load_config()
    spec = _spec(cfg["llm"])
    if not spec.api_key:
        env_name = cfg["llm"].get("api_key_env", "LLM_API_KEY")
        raise RuntimeError(
            f"未检测到有效的 API Key。请在项目根目录 .env 中设置 {env_name}（或通用的 LLM_API_KEY）。\n"
            f"当前 provider={spec.provider}, model={spec.model}, base_url={spec.base_url}"
        )
    return spec


def vlm_spec() -> ProviderSpec | None:
    """多模态模型未启用/未配置时返回 None，由调用方决定降级行为。"""
    cfg = load_config()
    section = cfg.get("vlm") or {}
    if not section.get("enabled") or not section.get("model"):
        return None
    spec = _spec(section)
    return spec if spec.api_key else None


def vlm_available() -> bool:
    return vlm_spec() is not None
