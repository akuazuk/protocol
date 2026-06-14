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


@lru_cache(maxsize=1)
def _protocol_id_to_local_path() -> dict[str, str]:
    """protocol_id → catalog local_path из json карточек."""
    out: dict[str, str] = {}
    ddir = _data_json_dir()
    if not ddir.is_dir():
        return out
    for path in sorted(ddir.glob("*.json")):
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        protocol_id = str(data.get("protocol_id") or path.stem)
        src = data.get("source") if isinstance(data.get("source"), dict) else {}
        lp = str((src or {}).get("local_path") or "").strip()
        if protocol_id and lp:
            out[protocol_id] = lp.replace("\\", "/")
    return out


def find_catalog_paths_by_icd_codes(codes: list[str], *, limit: int = 8) -> list[str]:
    """Пути PDF в каталоге по кодам МКБ (summary index → source.local_path)."""
    if not codes:
        return []
    id_map = _protocol_id_to_local_path()
    paths: list[str] = []
    seen: set[str] = set()
    for raw in codes:
        code = str(raw or "").strip()
        if not code:
            continue
        for pid, _cid in find_summary_refs_by_icd(code, limit=4):
            lp = id_map.get(pid)
            if not lp or lp in seen:
                continue
            seen.add(lp)
            paths.append(lp)
            if len(paths) >= limit:
                return paths
    return paths


def build_retrieval_prefilter_context(
    icd_codes: list[str] | None,
    category_slugs: list[str] | None,
    *,
    icd_path_limit: int = 24,
) -> dict[str, frozenset[str] | bool]:
    """Контекст pre-filter до embed rerank: пути PDF по МКБ + slug рубрик."""
    slugs = frozenset(
        s.strip()
        for s in (category_slugs or [])
        if isinstance(s, str) and s.strip()
    )
    paths: set[str] = set()
    if icd_codes:
        for p in find_catalog_paths_by_icd_codes(icd_codes, limit=icd_path_limit):
            paths.add(p.replace("\\", "/"))
    return {
        "active": bool(paths or slugs),
        "paths": frozenset(paths),
        "slugs": slugs,
    }


def _norm_catalog_path(p: str) -> str:
    return p.replace("\\", "/").strip().lower()


def catalog_path_matches_chunk(chunk_path: str, catalog_paths: frozenset[str]) -> bool:
    p = _norm_catalog_path(chunk_path)
    if not p or not catalog_paths:
        return False
    name = Path(p).name
    for raw in catalog_paths:
        np = _norm_catalog_path(raw)
        if not np:
            continue
        if p == np or p.endswith(np) or np.endswith(p) or name == Path(np).name:
            return True
    return False


def chunk_matches_retrieval_prefilter(
    ch: dict,
    *,
    catalog_paths: frozenset[str],
    category_slugs: frozenset[str],
    icd_norms: list[str],
) -> bool:
    """Чанк проходит pre-filter B2 (рубрика / МКБ-index / summary ICD)."""
    if catalog_paths and catalog_path_matches_chunk(str(ch.get("path") or ""), catalog_paths):
        return True
    cat = (ch.get("category") or "").strip()
    if category_slugs and cat in category_slugs:
        return True
    if not (ch.get("generated_from_summary") or ch.get("chunk_source") == "summary_chunks"):
        return False
    if category_slugs and cat in category_slugs:
        return True
    if icd_norms:
        for raw in ch.get("icd10_codes") or []:
            c = _norm_icd(str(raw)).lower()
            if not c:
                continue
            if any(c == n or c.startswith(n) or n.startswith(c) for n in icd_norms):
                return True
    return False


def clear_icd_summary_index_cache() -> None:
    _icd_to_summary_refs.cache_clear()
    _protocol_id_to_local_path.cache_clear()
