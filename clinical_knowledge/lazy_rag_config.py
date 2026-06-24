"""Feature flags и пути для lazy chunk store (manifest → disk → retrieve)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on", "y")


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def chunks_data_root() -> Path:
    raw = (os.environ.get("RAG_CHUNKS_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return ROOT


def startup_mode() -> str:
    """full = загрузка всех чанков; manifest = только manifest + метаданные."""
    mode = (os.environ.get("RAG_STARTUP_MODE") or "").strip().lower()
    if mode in ("manifest", "lazy", "lite"):
        return "manifest"
    if mode in ("full", "legacy"):
        return "full"
    if env_bool("RENDER", False):
        return "manifest"
    return "full"


def lazy_chunk_store_enabled() -> bool:
    if env_bool("RAG_LAZY_CHUNK_STORE", False):
        return True
    return startup_mode() == "manifest"


def lazy_retrieve_enabled() -> bool:
    if env_bool("RAG_LAZY_RETRIEVE", False):
        return True
    return startup_mode() == "manifest"


def forbid_full_corpus_retrieve() -> bool:
    if env_bool("RAG_FORBID_FULL_CORPUS_RETRIEVE", False):
        return True
    if env_bool("RAG_SEARCH_REQUIRE_ALLOWLIST_ON_RENDER", False):
        return True
    return env_bool("RENDER", False)


def path_lex_shards_enabled() -> bool:
    return env_bool("RAG_PATH_LEX_SHARDS", env_bool("RENDER", False))


def manifest_path() -> Path:
    raw = (os.environ.get("RAG_MANIFEST_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    default = chunks_data_root() / "corpus_path_manifest.jsonl"
    if default.is_file():
        return default
    return ROOT / "data/catalog/corpus_path_manifest.jsonl"


def lex_shards_dir() -> Path:
    raw = (os.environ.get("RAG_LEX_SHARDS_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    catalog = chunks_data_root() / "lex_shards"
    if catalog.is_dir():
        return catalog
    return ROOT / "data/catalog/lex_shards"


def chunk_cache_paths() -> int:
    return max(1, env_int("RAG_CHUNK_CACHE_PATHS", 16 if env_bool("RENDER", False) else 32))


def chunk_cache_max_chunks() -> int:
    return max(64, env_int("RAG_CHUNK_CACHE_MAX_CHUNKS", 2048 if env_bool("RENDER", False) else 4096))
