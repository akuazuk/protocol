"""Полнотекстовый reader протокола: абзацы для Study-режима навигатора.

Источник чтения: rich_chunks → source_text. Protocol Summary - только аннотации.
Не путать с protocol_brief (короткие «выводы» для Visit-режима).
"""
from __future__ import annotations

import re
from typing import Any

from clinical_knowledge.extract_quality import (
    clean_clinical_text,
    is_legal_admin_text,
    new_deduper,
    starts_like_sentence,
)
from clinical_knowledge.protocol_source_view import (
    _TYPE_TO_GROUP,
    _combine_title_text,
    _is_noise,
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z0-9«\"(])")
_SOFT_MIN = 180
_HARD_MIN = 280
_MAX_CHARS = 1200
_MAX_PER_SECTION = 16

_SECTION_LABELS: dict[str, str] = {
    "diagnosis": "Диагноз и классификация",
    "diagnostics": "Диагностика и обследования",
    "treatment": "Лечение и препараты",
    "followup": "Наблюдение, профилактика, маршрут",
}

_SECTION_ORDER = ("diagnosis", "diagnostics", "treatment", "followup")

# chunk_type → section_id (расширяет source_view; criteria встречается в тестах/корпусе)
_CHUNK_TYPE_MAP: dict[str, str] = {
    **_TYPE_TO_GROUP,
    "criteria": "diagnosis",
}

# ключи source_text.sections → id reader
_RAW_SECTION_MAP: dict[str, str] = {
    "classification": "diagnosis",
    "criteria": "diagnosis",
    "criteria_block": "diagnosis",
    "diagnostics": "diagnostics",
    "treatment": "treatment",
    "drug_list": "treatment",
    "pharmacotherapy": "treatment",
    "prevention": "followup",
    "routing": "followup",
    "dispensary": "followup",
    "rehabilitation": "followup",
    "rich": "",  # resolve via chunk_type
}


def _sentence_count(text: str) -> int:
    parts = [p for p in _SENT_SPLIT.split(text) if p.strip()]
    if parts:
        return len(parts)
    return 1 if text.strip() else 0


def _clip_paragraph(text: str, limit: int = _MAX_CHARS) -> str:
    t = clean_clinical_text(text)
    if len(t) <= limit:
        return t
    cut = t[:limit]
    # не резать посередине предложения, если есть точка в окне
    window = cut[ max(0, limit // 2) : ]
    m = list(re.finditer(r"[.!?»\"]\s+", window))
    if m:
        end = max(0, limit // 2) + m[-1].end()
        return t[:end].strip()
    # иначе по пробелу
    sp = cut.rfind(" ")
    if sp > limit // 2:
        return cut[:sp].rstrip() + "…"
    return cut.rstrip() + "…"


def _accept_paragraph(text: str) -> bool:
    t = clean_clinical_text(text)
    if len(t) < _SOFT_MIN:
        return False
    if is_legal_admin_text(t):
        return False
    if not starts_like_sentence(t):
        return False
    n_sent = _sentence_count(t)
    if len(t) >= _HARD_MIN:
        return n_sent >= 1
    # мягкий порог: нужно ≥2 предложения
    return n_sent >= 2


def _map_section(chunk_type: str | None, raw_key: str | None = None) -> str | None:
    ctype = str(chunk_type or "").strip().lower()
    if ctype in _CHUNK_TYPE_MAP:
        return _CHUNK_TYPE_MAP[ctype]
    key = str(raw_key or "").strip().lower()
    if key in _RAW_SECTION_MAP:
        mapped = _RAW_SECTION_MAP[key]
        return mapped or None
    return None


def _entities_from_block(block: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"drugs": [], "exams": []}
    for raw in block.get("drugs") or []:
        s = clean_clinical_text(str(raw))
        if 3 <= len(s) <= 48:
            out["drugs"].append(s)
    for key in ("lab_tests", "imaging", "procedures"):
        for raw in block.get(key) or []:
            s = clean_clinical_text(str(raw))
            if 3 <= len(s) <= 64:
                out["exams"].append(s)
    # unique preserve order
    for k in out:
        seen: set[str] = set()
        uniq: list[str] = []
        for x in out[k]:
            low = x.lower()
            if low in seen:
                continue
            seen.add(low)
            uniq.append(x)
        out[k] = uniq[:8]
    return out


def _iter_raw_blocks(doc: dict[str, Any]) -> list[tuple[str | None, dict[str, Any]]]:
    """(raw_section_key, block) из source_text / rich synthetic doc."""
    rows: list[tuple[str | None, dict[str, Any]]] = []
    sections = doc.get("sections") or {}
    if not isinstance(sections, dict):
        return rows
    for key, blocks in sections.items():
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if isinstance(block, dict):
                rows.append((str(key), block))
    return rows


def _doc_from_rich_chunks(chunks: list[dict[str, Any]], *, path: str) -> dict[str, Any] | None:
    """Синтетический doc: sections.rich с полным text (не lead из source_view)."""
    if not chunks:
        return None
    blocks: list[dict[str, Any]] = []
    title = ""
    protocol_id = ""
    for ch in chunks:
        text = str(ch.get("text") or "").strip()
        if not text:
            continue
        title = title or str(ch.get("protocol_title_normalized") or ch.get("protocol_title") or "")
        protocol_id = protocol_id or str(ch.get("doc_id") or "")
        blocks.append(
            {
                "chunk_type": ch.get("chunk_type") or ch.get("kind"),
                "section_title": ch.get("section_title") or ch.get("title"),
                "text": text,
                "page_from": ch.get("page_from") or ch.get("page"),
                "page_to": ch.get("page_to"),
                "drugs": ch.get("drugs") or [],
                "imaging": ch.get("imaging") or [],
                "lab_tests": ch.get("lab_tests") or [],
                "procedures": ch.get("procedures") or [],
            }
        )
    if not blocks:
        return None
    return {
        "path": path,
        "protocol_id": protocol_id,
        "title": title,
        "sections": {"rich": blocks},
    }


def _annotations_from_summary(catalog_path: str, section_id: str, text: str) -> list[dict[str, str]]:
    """Лёгкие аннотации из Summary card, если термин встречается в абзаце."""
    try:
        from clinical_knowledge.protocol_summary.nav import find_summary_by_catalog_path
    except Exception:
        return []
    summary = find_summary_by_catalog_path(catalog_path)
    if summary is None or not summary.conditions:
        return []
    cond = summary.conditions[0]
    low = text.lower()
    out: list[dict[str, str]] = []
    if section_id == "treatment":
        drugs = getattr(getattr(cond, "treatment", None), "drugs", None) or []
        for d in drugs[:12]:
            name = clean_clinical_text(getattr(d, "drug_name", None) or "")
            if len(name) < 3 or name.lower() not in low:
                continue
            bits = [name]
            dose = clean_clinical_text(getattr(d, "dose_text", None) or "")
            freq = clean_clinical_text(getattr(d, "frequency_text", None) or "")
            if dose:
                bits.append(dose)
            if freq:
                bits.append(freq)
            out.append({"kind": "dose", "label": " · ".join(bits)[:120]})
            if len(out) >= 4:
                break
    if section_id == "diagnostics":
        for ex in (getattr(cond, "required_exams", None) or [])[:12]:
            name = clean_clinical_text(getattr(ex, "name", None) or "")
            if len(name) < 4 or name.lower()[:24] not in low:
                continue
            level = str(getattr(ex, "requirement_level", "") or "")
            label = name if not level else f"{name} ({level})"
            out.append({"kind": "exam", "label": label[:120]})
            if len(out) >= 4:
                break
    return out


def build_protocol_reader(
    catalog_path: str,
    *,
    query: str = "",
    rich_chunks: list[dict[str, Any]] | None = None,
    max_per_section: int = _MAX_PER_SECTION,
) -> dict[str, Any]:
    """Собрать paragraph units для Study-навигатора."""
    from clinical_knowledge.protocol_links import normalize_protocol_path
    from clinical_knowledge.protocol_summary.source_text import resolve_protocol_source_text

    path = normalize_protocol_path(catalog_path) or (catalog_path or "").replace("\\", "/").strip()
    if not path:
        return {"available": False, "path": catalog_path, "paragraphs": [], "toc": [], "reason": "empty_path"}

    # L1: сырые rich_chunks (не view.lead). L2: source_text.sections с диска.
    source = "source_text"
    title = ""
    protocol_id: Any = None
    raw_rows: list[tuple[str | None, dict[str, Any]]] = []

    rich_doc = _doc_from_rich_chunks(list(rich_chunks or []), path=path)
    if rich_doc is not None:
        source = "rich_chunks"
        title = str(rich_doc.get("title") or "").strip()
        protocol_id = rich_doc.get("protocol_id")
        raw_rows = _iter_raw_blocks(rich_doc)

    if not raw_rows:
        doc = resolve_protocol_source_text(path, rich_chunks=None)
        if not doc.get("available"):
            return {
                "available": False,
                "path": path,
                "paragraphs": [],
                "toc": [],
                "title": str(doc.get("title") or title),
                "source": None,
                "reason": "no_source",
            }
        source = "source_text"
        title = title or str(doc.get("title") or "").strip()
        protocol_id = protocol_id or doc.get("protocol_id")
        raw_rows = _iter_raw_blocks(doc)

    buckets: dict[str, list[dict[str, Any]]] = {k: [] for k in _SECTION_ORDER}
    deduper = new_deduper()
    q_terms = [t for t in re.split(r"[^a-zA-Zа-яА-ЯёЁ0-9]+", (query or "").lower()) if len(t) >= 3]

    for raw_key, block in raw_rows:
        ctype = str(block.get("chunk_type") or block.get("kind") or "").strip().lower()
        section_id = _map_section(ctype, raw_key)
        if not section_id or section_id not in buckets:
            continue
        if ctype in ("body", "terms", "definitions", "protocol_overview", "table"):
            continue
        combined = _combine_title_text(block.get("section_title"), str(block.get("text") or ""))
        if not combined or _is_noise(combined, ctype):
            continue
        text = _clip_paragraph(combined, _MAX_CHARS)
        if not _accept_paragraph(text):
            continue
        if not deduper.accept(text):
            continue
        page = block.get("page_from") or block.get("page") or block.get("page_start")
        entities = _entities_from_block(block)
        annotations = _annotations_from_summary(path, section_id, text)
        score = 0
        if q_terms:
            low = text.lower()
            score = sum(1 for t in q_terms if t in low)
        buckets[section_id].append(
            {
                "section_id": section_id,
                "text": text,
                "page_start": page,
                "entities": entities,
                "annotations": annotations,
                "_score": score,
                "_len": len(text),
            }
        )

    paragraphs: list[dict[str, Any]] = []
    toc: list[dict[str, Any]] = []
    for section_id in _SECTION_ORDER:
        items = buckets[section_id]
        if not items:
            continue
        # query hits first, then longer paragraphs
        items.sort(key=lambda r: (-int(r.get("_score") or 0), -int(r.get("_len") or 0)))
        items = items[:max_per_section]
        # restore reading order by page then original-ish length
        items.sort(key=lambda r: (r.get("page_start") is None, r.get("page_start") or 0))
        toc.append(
            {
                "id": section_id,
                "label": _SECTION_LABELS[section_id],
                "count": len(items),
            }
        )
        for i, row in enumerate(items):
            paragraphs.append(
                {
                    "id": f"{section_id}-{i}",
                    "section_id": section_id,
                    "section_label": _SECTION_LABELS[section_id],
                    "text": row["text"],
                    "page_start": row.get("page_start"),
                    "entities": row.get("entities") or {"drugs": [], "exams": []},
                    "annotations": row.get("annotations") or [],
                }
            )

    return {
        "available": bool(paragraphs),
        "path": path,
        "protocol_id": protocol_id,
        "title": title,
        "source": source if paragraphs else None,
        "toc": toc,
        "paragraphs": paragraphs,
        "stats": {
            "paragraphs": len(paragraphs),
            "median_len": sorted(len(p["text"]) for p in paragraphs)[len(paragraphs) // 2] if paragraphs else 0,
            "with_page": sum(1 for p in paragraphs if p.get("page_start")),
            "ge_280": sum(1 for p in paragraphs if len(p["text"]) >= _HARD_MIN),
        },
        "reason": None if paragraphs else "no_paragraphs",
    }
