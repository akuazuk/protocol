"""Быстрый практический разбор без LLM — цитаты из rich-чанков + эвристическая матрица КЗ."""
from __future__ import annotations

import re
from typing import Any

from clinical_knowledge.rich_chunk_search import (
    _CHUNK_TYPE_LABELS,
    _LOW_SIGNAL_TYPES,
    chunk_type_multiplier,
    detect_query_intent,
)

_TYPE_TO_FIELD: dict[str, str] = {
    "diagnostics": "investigations",
    "criteria_block": "investigations",
    "classification": "diagnosis",
    "treatment": "treatment_methods",
    "pharmacotherapy": "medications",
    "drug_list": "medications",
    "prevention": "monitoring_followup",
    "dispensary": "monitoring_followup",
    "routing": "recommendations",
    "algorithm": "care_algorithms",
    "table": "investigations",
}


def _chunk_type(ch: dict[str, Any]) -> str:
    return (ch.get("chunk_type") or ch.get("kind") or "body").strip().lower()


def _lines_as_bullets(text: str, *, limit: int = 10) -> list[str]:
    out: list[str] = []
    for raw in (text or "").split("\n"):
        t = re.sub(r"^[\s\-•·\d.)]+", "", raw.strip())
        if len(t) < 14:
            continue
        if t not in out:
            out.append(t[:320])
        if len(out) >= limit:
            break
    if not out and (text or "").strip():
        out.append((text or "").strip()[:480])
    return out


def _score_chunk(ch: dict[str, Any], query: str, icd_codes: list[str] | None) -> float:
    mult = chunk_type_multiplier(query, ch, icd_codes=icd_codes)
    text = (ch.get("text") or "").lower()
    ql = (query or "").lower()
    overlap = 0
    for tok in re.findall(r"[а-яёa-z]{5,}", ql)[:10]:
        if tok in text:
            overlap += 1
    icd_boost = 0.0
    if icd_codes:
        icd_set = {c.upper() for c in icd_codes if c}
        for code in ch.get("icd10_codes") or []:
            if str(code).upper() in icd_set:
                icd_boost = 2.5
                break
    return mult + overlap * 0.35 + icd_boost


def _pick_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    icd_codes: list[str] | None,
    *,
    limit: int = 14,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, ch in enumerate(chunks):
        ctype = _chunk_type(ch)
        if ctype in _LOW_SIGNAL_TYPES and not (ch.get("icd10_codes") or []):
            continue
        text = (ch.get("text") or "").strip()
        if len(text) < 40:
            continue
        scored.append((_score_chunk(ch, query, icd_codes), idx, ch))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [ch for _s, _i, ch in scored[:limit]]


def build_lite_sections(
    chunks: list[dict[str, Any]],
    query: str,
    icd_codes: list[str] | None,
) -> list[dict[str, Any]]:
    """Короткие цитаты для UI (шаг 1 воронки)."""
    picked = _pick_chunks(chunks, query, icd_codes, limit=8)
    sections: list[dict[str, Any]] = []
    for ch in picked:
        ctype = _chunk_type(ch)
        label = (ch.get("section_title") or "").strip() or _CHUNK_TYPE_LABELS.get(ctype, ctype)
        text = (ch.get("text") or "").strip()
        sections.append(
            {
                "label": label[:120],
                "chunk_type": ctype,
                "text": text[:1200],
                "page_from": ch.get("page_from"),
                "page_to": ch.get("page_to"),
            }
        )
    return sections


def build_extraction_from_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    icd_codes: list[str] | None,
) -> dict[str, Any]:
    """Структура как у LLM extraction, но из rich-чанков."""
    picked = _pick_chunks(chunks, query, icd_codes, limit=16)
    fields: dict[str, list[str]] = {
        "investigations": [],
        "medications": [],
        "treatment_methods": [],
        "recommendations": [],
        "care_algorithms": [],
    }
    diagnosis_parts: list[str] = []
    monitoring: list[str] = []
    seen: set[str] = set()

    def _add(field: str, items: list[str]) -> None:
        for it in items:
            key = it[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            fields.setdefault(field, []).append(it)

    for ch in picked:
        ctype = _chunk_type(ch)
        bullets = _lines_as_bullets(ch.get("text") or "", limit=6)
        if ctype == "classification":
            diagnosis_parts.extend(bullets[:2])
            continue
        if ctype in ("prevention", "dispensary"):
            monitoring.extend(bullets[:3])
            continue
        target = _TYPE_TO_FIELD.get(ctype)
        if target == "diagnosis":
            diagnosis_parts.extend(bullets[:2])
        elif target:
            _add(target, bullets)

    extraction: dict[str, Any] = {
        "detailed": True,
        "investigations": fields.get("investigations", [])[:12],
        "medications": fields.get("medications", [])[:12],
        "treatment_methods": fields.get("treatment_methods", [])[:10],
        "recommendations": fields.get("recommendations", [])[:8],
        "care_algorithms": fields.get("care_algorithms", [])[:6],
    }
    if diagnosis_parts:
        extraction["diagnosis"] = diagnosis_parts[0][:500]
    if monitoring:
        extraction["monitoring_followup"] = monitoring[0][:400]
        extraction["monitoring_frequency"] = monitoring[0][:400]
    return extraction


def build_clinical_detail_lite(
    path: str,
    query: str,
    title: str,
    chunks: list[dict[str, Any]],
    icd_codes: list[str] | None = None,
) -> dict[str, Any]:
    has_rich = any(ch.get("rich_chunk") for ch in chunks)
    lite_sections = build_lite_sections(chunks, query, icd_codes) if chunks else []
    extraction = build_extraction_from_chunks(chunks, query, icd_codes) if chunks else {"detailed": False}
    score = 0.72
    if lite_sections:
        score = min(0.92, 0.62 + 0.03 * len(lite_sections))
    if icd_codes and any(
        str(c).upper() in " ".join((ch.get("text") or "") for ch in chunks[:20]).upper()
        for c in icd_codes
    ):
        score = min(0.95, score + 0.06)
    return {
        "path": path,
        "title": title,
        "source": "rich_chunks" if has_rich else "chunks_lite",
        "extraction": extraction,
        "lite_sections": lite_sections,
        "detail_match_score": round(score, 3),
        "llm_used": False,
    }
