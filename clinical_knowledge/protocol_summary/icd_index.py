"""Быстрый поиск Protocol Summary по МКБ без загрузки всего корпуса карточек."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _data_json_dir() -> Path:
    return ROOT / "data" / "protocol_summaries" / "json"


def _norm_icd(code: str) -> str:
    return re.sub(r"\s+", "", (code or "").upper().strip())


@lru_cache(maxsize=1)
def _icd_to_summary_refs() -> dict[str, list[tuple[str, str]]]:
    """МКБ -> [(protocol_id, condition_id), ...] из json без Pydantic."""
    out: dict[str, list[tuple[str, str]]] = {}
    ddir = _data_json_dir()
    if not ddir.is_dir():
        return out
    for path in sorted(ddir.glob("*.json")):
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        protocol_id = str(data.get("protocol_id") or path.stem)
        for cond in data.get("conditions") or []:
            if not isinstance(cond, dict):
                continue
            condition_id = str(cond.get("condition_id") or "")
            if not condition_id:
                continue
            ref = (protocol_id, condition_id)
            for raw in cond.get("icd10_codes") or []:
                code = _norm_icd(str(raw))
                if not code:
                    continue
                bucket = out.setdefault(code, [])
                if ref not in bucket:
                    bucket.append(ref)
    return out


def find_summary_refs_by_icd(icd10_code: str, *, limit: int = 4) -> list[tuple[str, str]]:
    """Возвращает до limit пар (protocol_id, condition_id) по точному или префиксному МКБ."""
    code = _norm_icd(icd10_code)
    if not code:
        return []
    idx = _icd_to_summary_refs()
    seen: set[tuple[str, str]] = set()
    found: list[tuple[str, str]] = []

    def _add(entries: list[tuple[str, str]]) -> None:
        for ref in entries:
            if ref in seen:
                continue
            seen.add(ref)
            found.append(ref)

    if code in idx:
        _add(idx[code])
    for key, entries in idx.items():
        if len(found) >= limit:
            break
        if key == code:
            continue
        if code.startswith(key) or key.startswith(code):
            _add(entries)
    return found[:limit]


def prewarm_icd_summary_index() -> int:
    """Прогрев индекса при старте сервера (опционально). Возвращает число кодов МКБ."""
    return len(_icd_to_summary_refs())


def clear_icd_summary_index_cache() -> None:
    _icd_to_summary_refs.cache_clear()
