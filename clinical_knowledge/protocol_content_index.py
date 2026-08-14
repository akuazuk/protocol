"""Поисковый текст КП: не только название, но содержание summary.

Индекс кладётся в data/catalog/protocol_content_index.json и попадает в образ GCE.
Если на диске есть Protocol Summary - они дополняют кэш.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data" / "catalog" / "protocol_content_index.json"
_MAX_TEXT = 1800
_TEXT_KEYS = frozenset(
    {
        "name",
        "synonyms",
        "abbreviations",
        "text",
        "quote",
        "title",
        "icd10_codes",
        "dose_text",
    }
)


def _norm_key(raw: str) -> str:
    text = (raw or "").strip().replace("\\", "/")
    if not text:
        return ""
    name = text.rsplit("/", 1)[-1].strip().lower()
    return name


def _collect_strings(obj: Any, acc: list[str], *, limit: int = 48, parent_key: str = "") -> None:
    if len(acc) >= limit:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            _collect_strings(value, acc, limit=limit, parent_key=str(key))
    elif isinstance(obj, list):
        for item in obj[:32]:
            _collect_strings(item, acc, limit=limit, parent_key=parent_key)
    elif isinstance(obj, str) and (parent_key in _TEXT_KEYS or parent_key == "icd10_codes"):
        text = re.sub(r"\s+", " ", obj).strip()
        if len(text) >= 4:
            acc.append(text[:240])


def content_text_from_summary(summary: Any) -> str:
    parts: list[str] = []
    source = getattr(summary, "source", None)
    title = str(getattr(source, "title", "") or "").strip()
    if title:
        parts.append(title)
    for cond in getattr(summary, "conditions", None) or []:
        name = str(getattr(cond, "name", "") or "").strip()
        if name:
            parts.append(name)
        for syn in getattr(cond, "synonyms", None) or []:
            if syn:
                parts.append(str(syn))
        for code in getattr(cond, "icd10_codes", None) or []:
            if code:
                parts.append(str(code))
    dumped = summary.model_dump() if hasattr(summary, "model_dump") else {}
    _collect_strings(dumped, parts)
    seen: set[str] = set()
    uniq: list[str] = []
    for item in parts:
        low = item.lower()
        if low in seen:
            continue
        seen.add(low)
        uniq.append(item)
    return re.sub(r"\s+", " ", " ".join(uniq)).strip()[:_MAX_TEXT]


def build_content_index_from_summaries() -> dict[str, str]:
    from clinical_knowledge.protocol_summary.loader import load_protocol_summaries

    index: dict[str, str] = {}
    for summary in load_protocol_summaries(usable_only=False):
        text = content_text_from_summary(summary)
        if not text:
            continue
        source = getattr(summary, "source", None)
        path = str(getattr(source, "local_path", "") or "").replace("\\", "/").strip()
        keys = [
            path.lower(),
            _norm_key(path),
            _norm_key(str(getattr(summary, "protocol_id", "") or "")),
        ]
        for key in keys:
            if not key:
                continue
            prev = index.get(key, "")
            if len(text) > len(prev):
                index[key] = text
    return index


def write_content_index(path: Path | None = None) -> Path:
    dest = path or INDEX_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = build_content_index_from_summaries()
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    return dest


@lru_cache(maxsize=1)
def _load_packaged_index() -> dict[str, str]:
    if not INDEX_PATH.is_file():
        return {}
    try:
        raw = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key).lower(): str(value) for key, value in raw.items() if key and value}


def _lookup_keys(card: dict[str, Any] | None) -> list[str]:
    card = card if isinstance(card, dict) else {}
    path = str(card.get("source_path") or "").replace("\\", "/").strip()
    keys = [
        path.lower(),
        _norm_key(path),
        _norm_key(str(card.get("protocol_id") or "")),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def content_text_for_card(card: dict[str, Any] | None) -> str:
    """Текст содержания КП для lexical match. Runtime - только упакованный JSON."""
    packaged = _load_packaged_index()
    chunks: list[str] = []
    seen: set[str] = set()
    for key in _lookup_keys(card):
        blob = packaged.get(key)
        if blob and blob not in seen:
            seen.add(blob)
            chunks.append(blob)
    if not chunks:
        return ""
    return re.sub(r"\s+", " ", " ".join(chunks)).strip()[:_MAX_TEXT]


def content_text_for_path(path: str) -> str:
    return content_text_for_card({"source_path": path})


def clear_content_index_cache() -> None:
    _load_packaged_index.cache_clear()


if __name__ == "__main__":
    dest = write_content_index()
    clear_content_index_cache()
    print(f"wrote {dest} keys={len(_load_packaged_index())}")
