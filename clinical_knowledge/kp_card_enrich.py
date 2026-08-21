"""Дополнить карточку КП кодами МКБ из содержания. Не выдумываем коды."""
from __future__ import annotations

import re
from typing import Any

_ICD_RE = re.compile(r"\b([A-TV-Z]\d{2}(?:\.\d{1,4})?)\b")


def icd_codes_from_text(text: str, *, limit: int = 40) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in _ICD_RE.finditer((text or "").upper()):
        code = match.group(1)
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
        if len(out) >= limit:
            break
    return out


def attach_icd_from_content(card: dict[str, Any] | None) -> dict[str, Any]:
    """Добавить на карточку коды, которые уже есть в тексте КП / content-index."""
    if not isinstance(card, dict):
        return {}
    try:
        from clinical_knowledge.protocol_content_index import content_text_for_card

        text = content_text_for_card(card)
    except Exception:  # noqa: BLE001
        text = ""
    found = icd_codes_from_text(text)
    if not found:
        return card
    current = [
        str(x).strip().upper()
        for x in list(card.get("icd10_all") or []) + list(card.get("icd10_primary") or [])
        if x
    ]
    have = set(current)
    extra = [code for code in found if code not in have]
    if not extra:
        return card
    mentions = [
        str(x).strip().upper()
        for x in (card.get("icd10_mentions") or [])
        if x
    ]
    have_mentions = set(mentions)
    mentions.extend(code for code in extra if code not in have_mentions)
    card["icd10_mentions"] = mentions[:80]
    card["icd10_all"] = (current + extra)[:80]
    if not card.get("icd10_primary"):
        card["icd10_primary"] = [code for code in extra if not code.startswith(("Y", "W", "V", "T"))][:8]
    return card
