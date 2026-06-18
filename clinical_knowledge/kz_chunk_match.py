"""Сопоставление пунктов КП с текстом КЗ (kz_match)."""
from __future__ import annotations

from typing import Any

from clinical_knowledge.semantic_rule_fallback import fuzzy_term_in_text


def match_kp_item_to_kz(
    item: str,
    kz_blob: str,
    *,
    entities: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Один пункт КП vs фрагмент КЗ."""
    text = (item or "").strip()
    blob = (kz_blob or "").strip()
    if not text:
        return {"kz_match": "missing", "kz_match_method": "none", "confidence": 0.0}
    if not blob:
        return {"kz_match": "missing", "kz_match_method": "none", "confidence": 0.0}

    ok, _, _ = fuzzy_term_in_text(blob, text)
    if ok:
        return {"kz_match": "found", "kz_match_method": "fuzzy", "confidence": 0.85}

    for group in (entities or {}).values():
        for ent in group or []:
            if len(ent) >= 4 and ent.lower() in blob.lower() and ent.lower() in text.lower():
                return {"kz_match": "found", "kz_match_method": "entity", "confidence": 0.75}

    head = text.split("—")[0].split("-")[0].strip()
    if len(head) >= 8:
        ok2, _, _ = fuzzy_term_in_text(blob, head)
        if ok2:
            return {"kz_match": "partial", "kz_match_method": "fuzzy_head", "confidence": 0.6}

    return {"kz_match": "missing", "kz_match_method": "fuzzy", "confidence": 0.0}


def match_kp_items(
    items: list[dict[str, Any]],
    kz_blob: str,
    *,
    chunk_entities: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Вернуть (found_items, missing_items) с kz_match метаданными."""
    found: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for it in items:
        row = dict(it)
        m = match_kp_item_to_kz(str(row.get("text") or ""), kz_blob, entities=chunk_entities)
        row.update(m)
        if m["kz_match"] in ("found", "partial"):
            found.append(row)
        else:
            missing.append(row)
    return found, missing


def best_chunk_for_items(
    chunks: list[dict[str, Any]],
    *,
    chunk_types: tuple[str, ...],
    icd_codes: list[str] | None = None,
) -> dict[str, Any] | None:
    """Лучший чанк для цитаты по типу и МКБ."""
    icd_set = {str(c).upper() for c in (icd_codes or []) if c}
    best: tuple[float, dict[str, Any]] | None = None
    for ch in chunks:
        ctype = (ch.get("chunk_type") or "").lower()
        if ctype not in chunk_types:
            continue
        tags = ch.get("tags") or {}
        if tags.get("signal") == "low" or tags.get("is_preamble"):
            continue
        score = 0.5
        if tags.get("obligation") == "required":
            score += 0.4
        if tags.get("signal") == "high":
            score += 0.2
        ch_icd = set(str(c).upper() for c in (ch.get("icd10_codes") or []))
        if icd_set & ch_icd:
            score += 0.5
        weights = tags.get("icd10_weights") or ch.get("icd10_weights") or {}
        for code in icd_set:
            try:
                score += float(weights.get(code) or 0) * 0.3
            except (TypeError, ValueError):
                pass
        if best is None or score > best[0]:
            best = (score, ch)
    return best[1] if best else None
