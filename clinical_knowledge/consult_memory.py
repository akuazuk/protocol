"""Ограничения RAM для проверки КЗ на Render (2 GiB)."""
from __future__ import annotations

import os
from typing import Any, Callable

GetChunksFn = Callable[[str], list[dict[str, Any]]]


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


def consult_forbid_full_corpus() -> bool:
    """Запрет retrieve() по всему корпусу (~55k чанков) во время КЗ."""
    if env_bool("CONSULT_REVIEW_FORBID_FULL_CORPUS", False):
        return True
    return env_bool("RENDER", False)


def consult_max_paths() -> int:
    return max(1, min(8, env_int("CONSULT_REVIEW_MAX_PROTOCOL_PATHS", 4)))


def consult_max_chunks_per_path() -> int:
    return max(4, min(64, env_int("CONSULT_RICH_CHUNKS_MAX_PER_PATH", 24)))


def consult_max_chunk_chars() -> int:
    return max(512, min(8192, env_int("CONSULT_RICH_CHUNK_MAX_CHARS", 2048)))


def cap_chunks_for_consult(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Урезать rich-чанки одного PDF для alignment/L2-lite (не держать сотни чанков в RAM)."""
    max_n = consult_max_chunks_per_path()
    max_chars = consult_max_chunk_chars()
    keep_keys = (
        "path",
        "source_path",
        "text",
        "lex_text",
        "chunk_id",
        "chunk_type",
        "kind",
        "section_title",
        "section_path",
        "page_from",
        "page_to",
        "icd10_codes",
        "icd10_weights",
        "chunk_index",
        "rich_chunk",
        "specialty_slug",
    )
    out: list[dict[str, Any]] = []
    for ch in chunks[:max_n]:
        if not isinstance(ch, dict):
            continue
        slim: dict[str, Any] = {}
        for k in keep_keys:
            if k in ch and ch[k] is not None:
                slim[k] = ch[k]
        txt = str(slim.get("text") or slim.get("lex_text") or "")
        if len(txt) > max_chars:
            slim["text"] = txt[:max_chars]
        if slim.get("lex_text") and len(str(slim["lex_text"])) > max_chars:
            slim["lex_text"] = str(slim["lex_text"])[:max_chars]
        if slim.get("text") or slim.get("lex_text"):
            out.append(slim)
    return out


def get_rich_chunks_for_consult(path: str, get_chunks: GetChunksFn) -> list[dict[str, Any]]:
    return cap_chunks_for_consult(list(get_chunks(path) or []))
