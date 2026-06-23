"""Human-readable match tier and explanation for protocol search UI."""

from __future__ import annotations

from typing import Any

from icd_mkb import normalize_icd_code as normalize_code


def _icd_strength(pr: dict[str, Any]) -> float:
    try:
        return float(pr.get("icd_match_strength") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def compute_match_tier(pr: dict[str, Any], icd_codes: list[str] | None = None) -> str:
    """icd_primary | icd_secondary | text_only | manual_check."""
    strength = _icd_strength(pr)
    matched = [normalize_code(str(c)) for c in (pr.get("matched_icd_codes") or []) if normalize_code(str(c))]
    query_icd = [normalize_code(str(c)) for c in (icd_codes or []) if normalize_code(str(c))]
    if strength >= 70.0 or (matched and query_icd and any(m in query_icd or any(m.startswith(q) or q.startswith(m) for q in query_icd) for m in matched)):
        if strength >= 70.0 or any(m in query_icd for m in matched):
            return "icd_primary"
        return "icd_secondary"
    if matched or strength >= 40.0:
        return "icd_secondary"
    try:
        conf = float(pr.get("confidence_score") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    if conf >= 0.55:
        return "text_only"
    return "manual_check"


def compute_match_explain_ru(pr: dict[str, Any], icd_codes: list[str] | None = None) -> str:
    """One-line Russian explanation for doctors."""
    existing = str(pr.get("match_reason") or "").strip()
    matched = [str(c) for c in (pr.get("matched_icd_codes") or []) if c][:3]
    query_icd = [str(c) for c in (icd_codes or []) if c][:3]
    tier = pr.get("match_tier") or compute_match_tier(pr, icd_codes)
    if tier == "icd_primary" and matched:
        code = matched[0]
        if query_icd and code not in query_icd:
            return f"Код {query_icd[0]} → {code} в протоколе"
        return f"Код {code} в разделе классификации протокола"
    if tier == "icd_secondary" and matched:
        return f"Связанные коды МКБ: {', '.join(matched[:2])}"
    if existing and not existing.startswith("МКБ"):
        return existing[:90]
    if existing:
        return existing[:90]
    section = str(pr.get("section_title") or "").strip()
    if section:
        return f"Фрагмент: {section[:60]}"
    if tier == "text_only":
        return "Подобрано по тексту жалобы и рубрике"
    return "Слабое совпадение - сверьте с клиникой и каталогом МЗ"


def enrich_protocol_match_ui(
    protocols: list[dict[str, Any]],
    icd_codes: list[str] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pr in protocols:
        if not isinstance(pr, dict):
            continue
        row = dict(pr)
        row["match_tier"] = compute_match_tier(row, icd_codes)
        row["match_explain_ru"] = compute_match_explain_ru(row, icd_codes)
        out.append(row)
    return out
