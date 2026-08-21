"""Паспорт МКБ карточки КП: primary vs mentions из тела PDF."""
from __future__ import annotations

from typing import Any

_MENTION_LETTERS = frozenset("YWVT")


def _norm_code(code: str) -> str:
    return (code or "").upper().strip()


def _is_mention_only(code: str) -> bool:
    text = _norm_code(code)
    return bool(text) and text[0] in _MENTION_LETTERS


def _uniq(codes: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in codes:
        code = _norm_code(str(raw or ""))
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def apply_icd_passport(
    card: dict[str, Any] | None,
    catalog_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Primary = паспорт/классификация. Mentions = тело PDF, suggest их не видит."""
    if not isinstance(card, dict):
        return {}
    old_primary = [str(x) for x in (card.get("icd10_primary") or []) if x]
    old_all = [str(x) for x in (card.get("icd10_all") or []) if x]
    old_mentions = [str(x) for x in (card.get("icd10_mentions") or []) if x]
    catalog = catalog_row if isinstance(catalog_row, dict) else {}
    cat_primary = [
        str(x)
        for x in (catalog.get("icd10_primary") or [])
        if x and not _is_mention_only(str(x))
    ]
    cat_all = [str(x) for x in (catalog.get("icd10_all") or []) if x]
    if cat_primary:
        primary = _uniq(cat_primary)
    else:
        primary = _uniq([code for code in old_primary if not _is_mention_only(code)])
    primary_set = set(primary)
    mentions = _uniq(
        [
            code
            for code in old_mentions + old_all + old_primary + cat_all
            if _norm_code(code) not in primary_set
        ]
    )
    card["icd10_primary"] = primary[:16]
    card["icd10_mentions"] = mentions[:80]
    card["icd10_all"] = _uniq(primary + mentions)[:80]
    return card


def suggest_icd_codes(card: dict[str, Any] | None) -> list[str]:
    """Коды для подбора: primary + не-омнибус all без внешних причин (Y/W/V/T)."""
    if not isinstance(card, dict):
        return []
    primary = [code for code in (card.get("icd10_primary") or []) if not _is_mention_only(str(code))]
    extra: list[str] = []
    try:
        from clinical_knowledge.kp_validity import looks_omnibus

        if not looks_omnibus(card):
            extra = [
                str(code)
                for code in (card.get("icd10_all") or [])
                if code and not _is_mention_only(str(code))
            ]
    except Exception:  # noqa: BLE001
        extra = []
    return _uniq([str(x) for x in primary + extra if x])
