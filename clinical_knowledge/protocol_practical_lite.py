"""Быстрый практический разбор без LLM - обследование и лечение из rich-чанков."""
from __future__ import annotations

import re
from typing import Any

from clinical_knowledge.rich_chunk_search import (
    _CHUNK_TYPE_LABELS,
    _LOW_SIGNAL_TYPES,
    chunk_type_multiplier,
    detect_query_intent,
)

SECTION_CHUNK_TYPES: dict[str, tuple[str, ...]] = {
    "investigations": ("diagnostics", "criteria_block", "table", "protocol_overview"),
    "medications": ("pharmacotherapy", "drug_list"),
    "treatment_methods": ("treatment", "pharmacotherapy"),
    "monitoring_frequency": ("prevention", "dispensary"),
    "care_algorithms": ("algorithm",),
}

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
    "protocol_overview": "investigations",
}

_DIAG_LINE = re.compile(
    r"анализ|исследован|обслед|узи|ультразвук|рентген|кт\b|мрт|экг|эхо|бакпосев|"
    r"мазок|биохим|гемат|коагул|бронхоскоп|спиромет|томограф|консультац|осмотр|"
    r"пункци|биопси|пцр|ифа|посев|оак\b|оам\b",
    re.I,
)
_MED_LINE = re.compile(
    r"мг\b|мкг|мл\b|таблет|капсул|суспенз|инъекц|внутримышеч|per\s*os|"
    r"антибиот|противовирус|нпвс|ингаляц|препарат|сут\b|раз\s+в\s+день|"
    r"в\s+сутки|доза|мг/кг",
    re.I,
)
_TREAT_LINE = re.compile(
    r"лечен|терапи|назнач|операц|хирург|госпитал|амбулатор|постельн|режим|"
    r"физиотерап|реабилит|ингаляц|оксиген|бронхолит",
    re.I,
)
_NOISE_LINE = re.compile(
    r"постановлен|утвержд|министерств|настоящ(ий|его)\s+клиническ|"
    r"термины\s+и\s+определен|сокращен|список\s+литератур",
    re.I,
)


def _chunk_type(ch: dict[str, Any]) -> str:
    return (ch.get("chunk_type") or ch.get("kind") or "body").strip().lower()


def _is_noise_line(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 12:
        return True
    if _NOISE_LINE.search(t):
        return True
    if t.isupper() and len(t) > 40:
        return True
    return False


def _lines_as_bullets(text: str, *, limit: int = 10) -> list[str]:
    out: list[str] = []
    for raw in (text or "").split("\n"):
        t = re.sub(r"^[\s\-•·\d.)]+", "", raw.strip())
        if len(t) < 14 or _is_noise_line(t):
            continue
        if t not in out:
            out.append(t[:320])
        if len(out) >= limit:
            break
    if not out and (text or "").strip() and not _is_noise_line(text):
        out.append((text or "").strip()[:480])
    return out


def _filter_lines_for_field(lines: list[str], field: str) -> list[str]:
    if field == "investigations":
        return [ln for ln in lines if _DIAG_LINE.search(ln)] or lines[:4]
    if field == "medications":
        return [ln for ln in lines if _MED_LINE.search(ln)] or lines[:4]
    if field == "treatment_methods":
        med = [ln for ln in lines if _MED_LINE.search(ln)]
        treat = [ln for ln in lines if _TREAT_LINE.search(ln) and ln not in med]
        return treat or [ln for ln in lines if ln not in med][:4]
    return lines


def _score_chunk(ch: dict[str, Any], query: str, icd_codes: list[str] | None) -> float:
    mult = chunk_type_multiplier(query, ch, icd_codes=icd_codes)
    text = (ch.get("text") or "").lower()
    ql = (query or "").lower()
    overlap = 0
    for tok in re.findall(r"[а-яёa-z]{5,}", ql)[:12]:
        if tok in text:
            overlap += 1
    icd_boost = 0.0
    if icd_codes:
        icd_set = {c.upper() for c in icd_codes if c}
        for code in ch.get("icd10_codes") or []:
            if str(code).upper() in icd_set:
                icd_boost = 2.5
                break
        weights = ch.get("icd10_weights") or {}
        if weights and any(str(k).upper() in icd_set for k in weights):
            icd_boost = max(icd_boost, 1.8)
    ctype = _chunk_type(ch)
    type_boost = 0.0
    if ctype in ("diagnostics", "criteria_block", "pharmacotherapy", "drug_list", "treatment"):
        type_boost = 1.2
    elif ctype == "protocol_overview":
        type_boost = 0.4
    return mult + overlap * 0.4 + icd_boost + type_boost


def _pick_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    icd_codes: list[str] | None,
    *,
    limit: int = 14,
    chunk_types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, int, dict[str, Any]]] = []
    allowed = set(chunk_types) if chunk_types else None
    for idx, ch in enumerate(chunks):
        ctype = _chunk_type(ch)
        if allowed and ctype not in allowed:
            continue
        if ctype in _LOW_SIGNAL_TYPES and not (ch.get("icd10_codes") or ch.get("icd10_weights")):
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
    picked = _pick_chunks(chunks, query, icd_codes, limit=10)
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
    """Структура как у LLM extraction, но из rich-чанков с фокусом на диагностику и лечение."""
    diag_chunks = _pick_chunks(
        chunks, query, icd_codes, limit=10, chunk_types=("diagnostics", "criteria_block", "table")
    )
    med_chunks = _pick_chunks(
        chunks, query, icd_codes, limit=8, chunk_types=("pharmacotherapy", "drug_list")
    )
    treat_chunks = _pick_chunks(chunks, query, icd_codes, limit=6, chunk_types=("treatment",))
    algo_chunks = _pick_chunks(chunks, query, icd_codes, limit=4, chunk_types=("algorithm",))
    monitor_chunks = _pick_chunks(
        chunks, query, icd_codes, limit=4, chunk_types=("prevention", "dispensary")
    )

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
        filtered = _filter_lines_for_field(items, field)
        for it in filtered:
            key = it[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            fields.setdefault(field, []).append(it)

    for ch in diag_chunks:
        _add("investigations", _lines_as_bullets(ch.get("text") or "", limit=8))
    for ch in med_chunks:
        _add("medications", _lines_as_bullets(ch.get("text") or "", limit=8))
    for ch in treat_chunks:
        _add("treatment_methods", _lines_as_bullets(ch.get("text") or "", limit=6))
    for ch in algo_chunks:
        _add("care_algorithms", _lines_as_bullets(ch.get("text") or "", limit=5))
    for ch in monitor_chunks:
        monitoring.extend(_lines_as_bullets(ch.get("text") or "", limit=3))

    for ch in _pick_chunks(chunks, query, icd_codes, limit=6):
        ctype = _chunk_type(ch)
        if ctype == "classification":
            diagnosis_parts.extend(_lines_as_bullets(ch.get("text") or "", limit=2))

    if not fields["investigations"]:
        for ch in _pick_chunks(chunks, query, icd_codes, limit=8):
            if _chunk_type(ch) == "protocol_overview":
                _add("investigations", _lines_as_bullets(ch.get("text") or "", limit=4))

    extraction: dict[str, Any] = {
        "detailed": True,
        "investigations": fields.get("investigations", [])[:14],
        "medications": fields.get("medications", [])[:14],
        "treatment_methods": fields.get("treatment_methods", [])[:12],
        "recommendations": fields.get("recommendations", [])[:6],
        "care_algorithms": fields.get("care_algorithms", [])[:8],
    }
    if diagnosis_parts:
        extraction["diagnosis"] = diagnosis_parts[0][:500]
    if monitoring:
        extraction["monitoring_followup"] = monitoring[0][:400]
        extraction["monitoring_frequency"] = monitoring[0][:400]
    return extraction


def _cites_from_sections(
    lite_sections: list[dict[str, Any]],
    chunk_types: tuple[str, ...],
    *,
    limit: int = 2,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sec in lite_sections:
        if (sec.get("chunk_type") or "") not in chunk_types:
            continue
        out.append(sec)
        if len(out) >= limit:
            break
    return out


def infer_query_clinical_focus(query: str) -> str:
    ql = (query or "").lower()
    if re.search(r"препарат|антибиот|таблет|доз|мг\b|назнач", ql):
        return "treatment"
    if re.search(r"обслед|анализ|узи|кт\b|рентген|диагност", ql):
        return "diagnostics"
    if re.search(r"лечен|терапи|операц", ql):
        return "treatment"
    return "diagnostics"


def build_clinical_blocks(
    extraction: dict[str, Any],
    lite_sections: list[dict[str, Any]],
    query: str,
) -> dict[str, Any]:
    """Структура для UI: диагностика и лечение с цитатами из протокола."""
    ex = extraction or {}
    diag_types = ("diagnostics", "criteria_block", "table", "protocol_overview")
    treat_types = ("treatment", "pharmacotherapy", "drug_list")
    return {
        "query_focus": infer_query_clinical_focus(query),
        "diagnostics": {
            "title": "Обследование и диагностика",
            "items": list(ex.get("investigations") or [])[:14],
            "cites": _cites_from_sections(lite_sections, diag_types, limit=2),
        },
        "treatment": {
            "title": "Лечение и препараты",
            "methods": list(ex.get("treatment_methods") or [])[:10],
            "medications": list(ex.get("medications") or [])[:12],
            "cites": _cites_from_sections(lite_sections, treat_types, limit=2),
        },
        "monitoring": {
            "text": (ex.get("monitoring_frequency") or ex.get("monitoring_followup") or "")[:400],
        },
        "algorithms": list(ex.get("care_algorithms") or [])[:6],
    }


def normalize_practical_section(raw: str | None) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    x = raw.strip().lower()
    if x == "monitoring":
        x = "monitoring_frequency"
    if x == "algorithms":
        x = "care_algorithms"
    if x in SECTION_CHUNK_TYPES:
        return x
    return None


def build_practical_section(
    path: str,
    query: str,
    title: str,
    chunks: list[dict[str, Any]],
    section: str,
    icd_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Один раздел практического разбора из rich-чанков (без LLM)."""
    sec = normalize_practical_section(section)
    if not sec:
        raise ValueError(f"Неизвестный раздел: {section}")
    allowed = set(SECTION_CHUNK_TYPES[sec])
    filtered = [ch for ch in chunks if _chunk_type(ch) in allowed]
    picked = _pick_chunks(filtered, query, icd_codes, limit=10)
    items: list[str] = []
    cites: list[dict[str, Any]] = []
    seen: set[str] = set()
    field = {
        "investigations": "investigations",
        "medications": "medications",
        "treatment_methods": "treatment_methods",
        "monitoring_frequency": "monitoring_frequency",
        "care_algorithms": "care_algorithms",
    }.get(sec, sec)
    for ch in picked:
        ctype = _chunk_type(ch)
        label = (ch.get("section_title") or "").strip() or _CHUNK_TYPE_LABELS.get(ctype, ctype)
        text = (ch.get("text") or "").strip()
        if text:
            cites.append(
                {
                    "label": label[:120],
                    "chunk_type": ctype,
                    "text": text[:1200],
                    "page_from": ch.get("page_from"),
                    "page_to": ch.get("page_to"),
                }
            )
        for bullet in _filter_lines_for_field(_lines_as_bullets(text, limit=8), field):
            key = bullet[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(bullet)
            if len(items) >= 12:
                break
        if len(items) >= 12:
            break
    has_rich = any(ch.get("rich_chunk") for ch in chunks)
    return {
        "path": path,
        "title": title,
        "section": sec,
        "items": items[:12],
        "cites": cites[:3],
        "source": "rich_chunks" if has_rich else "chunks_lite",
    }


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
    clinical_blocks = build_clinical_blocks(extraction, lite_sections, query) if chunks else {}
    score = 0.72
    n_useful = len(extraction.get("investigations") or []) + len(extraction.get("medications") or [])
    if lite_sections:
        score = min(0.92, 0.58 + 0.03 * len(lite_sections) + 0.02 * min(n_useful, 8))
    if icd_codes and any(
        str(c).upper() in " ".join((ch.get("text") or "") for ch in chunks[:24]).upper()
        for c in icd_codes
    ):
        score = min(0.95, score + 0.06)
    return {
        "path": path,
        "title": title,
        "source": "rich_chunks" if has_rich else "chunks_lite",
        "extraction": extraction,
        "clinical_blocks": clinical_blocks,
        "lite_sections": lite_sections,
        "detail_match_score": round(score, 3),
        "llm_used": False,
    }
