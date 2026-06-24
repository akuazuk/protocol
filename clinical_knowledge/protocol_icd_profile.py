"""Профиль КП по МКБ из rich-чанков (обследование, лечение, наблюдение)."""
from __future__ import annotations

import re
from typing import Any, Callable

from clinical_knowledge.protocol_practical_lite import (
    _chunk_type,
    _lines_as_bullets,
    _pick_chunks,
    build_extraction_from_chunks,
)

GetChunksFn = Callable[[str], list[dict[str, Any]]]


def _icd_match_chunk(ch: dict[str, Any], icd_codes: list[str]) -> bool:
    if not icd_codes:
        return True
    icd_set = {c.upper() for c in icd_codes if c}
    for code in ch.get("icd10_codes") or []:
        if str(code).upper() in icd_set:
            return True
    weights = ch.get("icd10_weights") or {}
    return any(str(k).upper() in icd_set for k in weights)


def _chunk_cite(ch: dict[str, Any], path: str) -> dict[str, Any]:
    text = (ch.get("text") or "").strip()
    return {
        "path": path,
        "chunk_id": ch.get("chunk_id") or ch.get("id"),
        "chunk_type": _chunk_type(ch),
        "section_title": (ch.get("section_title") or "").strip(),
        "page_from": ch.get("page_from"),
        "page_to": ch.get("page_to"),
        "text": text[:600],
    }


def build_protocol_icd_profile(
    chunks: list[dict[str, Any]],
    icd_codes: list[str],
    *,
    path: str = "",
    query: str = "",
) -> dict[str, Any]:
    """Извлечь обследования/лечение/наблюдение из чанков одного PDF."""
    if not chunks:
        return {
            "path": path,
            "diagnostics": [],
            "medications": [],
            "treatment": [],
            "monitoring": [],
            "cites": [],
        }

    filtered = [ch for ch in chunks if _icd_match_chunk(ch, icd_codes)] or chunks
    q = query or " ".join(icd_codes)
    extraction = build_extraction_from_chunks(filtered, q, icd_codes)

    diag_chunks = _pick_chunks(
        filtered, q, icd_codes, limit=8, chunk_types=("diagnostics", "criteria_block", "table")
    )
    treat_chunks = _pick_chunks(
        filtered, q, icd_codes, limit=6, chunk_types=("treatment", "pharmacotherapy", "drug_list")
    )
    mon_chunks = _pick_chunks(
        filtered, q, icd_codes, limit=4, chunk_types=("prevention", "dispensary")
    )

    cites: list[dict[str, Any]] = []
    for ch in diag_chunks[:2] + treat_chunks[:2] + mon_chunks[:1]:
        cites.append(_chunk_cite(ch, path))

    monitoring: list[str] = []
    for ch in mon_chunks:
        monitoring.extend(_lines_as_bullets(ch.get("text") or "", limit=4))

    return {
        "path": path,
        "diagnostics": list(extraction.get("investigations") or [])[:16],
        "medications": list(extraction.get("medications") or [])[:14],
        "treatment": list(extraction.get("treatment_methods") or [])[:12],
        "monitoring": monitoring[:8],
        "cites": cites,
    }


def merge_protocol_profiles(
    paths: list[str],
    icd_codes: list[str],
    get_chunks: GetChunksFn,
    *,
    query: str = "",
) -> dict[str, Any]:
    """Агрегировать профили по нескольким PDF протоколам."""
    profiles: list[dict[str, Any]] = []
    seen_diag: set[str] = set()
    seen_med: set[str] = set()
    seen_treat: set[str] = set()
    seen_mon: set[str] = set()
    merged = {
        "paths": [],
        "diagnostics": [],
        "medications": [],
        "treatment": [],
        "monitoring": [],
        "cites": [],
    }

    for path in paths:
        chunks = get_chunks(path) or []
        try:
            from clinical_knowledge.consult_memory import cap_chunks_for_consult, consult_forbid_full_corpus

            if consult_forbid_full_corpus():
                chunks = cap_chunks_for_consult(chunks)
        except Exception:
            pass
        if not chunks:
            continue
        prof = build_protocol_icd_profile(chunks, icd_codes, path=path, query=query)
        profiles.append(prof)
        merged["paths"].append(path)

        def _add_unique(bucket: str, items: list[str], seen: set[str]) -> None:
            for it in items:
                key = re.sub(r"\s+", " ", (it or "").lower())[:80]
                if not key or key in seen:
                    continue
                seen.add(key)
                merged[bucket].append(it)

        _add_unique("diagnostics", prof.get("diagnostics") or [], seen_diag)
        _add_unique("medications", prof.get("medications") or [], seen_med)
        _add_unique("treatment", prof.get("treatment") or [], seen_treat)
        _add_unique("monitoring", prof.get("monitoring") or [], seen_mon)

        for cite in prof.get("cites") or []:
            if len(merged["cites"]) < 6:
                merged["cites"].append(cite)

    merged["profiles"] = profiles
    return merged
