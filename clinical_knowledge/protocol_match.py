"""Подбор карточек протоколов по фактам из КЗ."""
from __future__ import annotations

from typing import Any

from .loader import load_protocol_cards_registry


def _icd_root(code: str) -> str:
    c = (code or "").upper().strip()
    return c[:3] if len(c) >= 3 else c


def _population_match(card_pop: str, consult_audience: str | None) -> float:
    cp = (card_pop or "any").lower()
    ca = (consult_audience or "").lower()
    if not ca or cp == "any":
        return 1.0
    if cp == ca:
        return 1.0
    return 0.0


def match_protocol_cards(
    consult_facts: dict[str, Any],
    *,
    specialty_slug: str | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Ранжированный список protocol_id по МКБ, популяции, нозологии."""
    cards = load_protocol_cards_registry()
    if specialty_slug:
        cards = [c for c in cards if c.get("specialty_slug") == specialty_slug]

    ctx = consult_facts.get("patient_context") or {}
    cons = consult_facts.get("consultation") or {}
    icd_list = [str(x).upper() for x in (cons.get("icd10") or [])]
    icd_roots = {_icd_root(c) for c in icd_list}
    audience = ctx.get("adult_or_child")
    hints = set(cons.get("conditions_hint") or [])

    scored: list[tuple[float, dict[str, Any]]] = []
    for card in cards:
        score = 0.0
        card_icd = [str(x).upper() for x in (card.get("icd10_all") or card.get("icd10_primary") or [])]
        card_roots = {_icd_root(c) for c in card_icd}

        icd_overlap = icd_roots & card_roots
        if icd_overlap:
            score += 40 + 5 * len(icd_overlap)
        elif icd_list and card_icd:
            for c in icd_list:
                for cc in card_icd:
                    if c.startswith(_icd_root(cc)) or cc.startswith(_icd_root(c)):
                        score += 25
                        break

        pop_mult = _population_match(str(card.get("population") or "any"), audience)
        if pop_mult == 0:
            score -= 50
        elif pop_mult == 1.0 and audience:
            score += 15

        title_low = (card.get("title") or "").lower()
        path_low = (card.get("source_path") or "").lower()
        blob = title_low + " " + path_low
        if "gerd" in hints or any(c.startswith("K21") for c in icd_list):
            if any(x in blob for x in ("гэрб", "рефлюкс", "пищевод", "желудк", "двенадцат")):
                score += 35
        if "gastritis" in hints and "гастрит" in blob:
            score += 25
        if "peptic_ulcer" in hints and "язв" in blob:
            score += 25
        if "crohn" in hints and ("крон" in blob or "k50" in blob):
            score += 30
        if "ulcerative_colitis" in hints and ("колит" in blob or "k51" in blob):
            score += 30
        if "celiac" in hints and "целиак" in blob:
            score += 30
        if "acute_pancreatitis" in hints and "панкреат" in blob:
            score += 28
        if "acute_appendicitis" in hints and "аппендицит" in blob:
            score += 28

        if (card.get("status") or "active") != "active":
            score -= 20

        if score > 0:
            scored.append((score, card))

    scored.sort(key=lambda x: (-x[0], x[1].get("protocol_id") or ""))
    out: list[dict[str, Any]] = []
    for sc, card in scored[:limit]:
        out.append(
            {
                "protocol_id": card.get("protocol_id"),
                "title": card.get("title"),
                "source_path": card.get("source_path"),
                "population": card.get("population"),
                "icd10_primary": card.get("icd10_primary"),
                "match_score": round(sc, 2),
                "approval": card.get("approval"),
            }
        )
    return out
