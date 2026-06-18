"""Детальный match_score с breakdown для UI и аудита."""
from __future__ import annotations

from typing import Any

from clinical_knowledge.protocol_match import (
    _card_spine_bladder_relevance,
    _card_venous_relevance,
    _diag_text_overlap,
    _is_spine_icd,
    _is_venous_icd,
    _population_match,
    _WEIGHT_COMPLAINTS,
    _WEIGHT_DEMO,
    _WEIGHT_DIAG_TEXT,
    _WEIGHT_EXAMS,
    _WEIGHT_ICD,
    _WEIGHT_POPULATION,
    _WEIGHT_SPECIALTY,
)
from clinical_knowledge.protocol_pick_filters import clinical_relevance_multiplier, icd_fit_for_card
from clinical_knowledge.diagnosis_icd import is_symptom_code


def _icd_root(code: str) -> str:
    c = (code or "").upper().strip()
    return c[:3] if len(c) >= 3 else c


def compute_match_detail(
    card: dict[str, Any],
    *,
    icd_list: list[str],
    audience: str | None,
    hints: set[str],
    specialty_slug: str | None,
    diag_text: str,
    complaints: list[str],
    performed_exams: list[str],
) -> dict[str, Any]:
    """Score 0–100 + breakdown + icd_fit + risk_flags."""
    from clinical_knowledge.condition_registry import score_card_for_hint
    from clinical_knowledge.protocol_pick_filters import is_administrative_protocol

    risk_flags: list[str] = []
    if is_administrative_protocol(card):
        risk_flags.append("admin_order")
        return {
            "match_score": 0.0,
            "match_breakdown": {},
            "icd_fit": [],
            "icd_fit_label": "",
            "pick_risk_flags": risk_flags,
            "pick_reason_ru": "Приказ об утверждении — не клинический эталон",
            "rejected": True,
        }

    card_icd = [str(x).upper() for x in (card.get("icd10_all") or card.get("icd10_primary") or [])]
    icd_roots = {_icd_root(c) for c in icd_list if not is_symptom_code(c)}
    card_roots = {_icd_root(c) for c in card_icd}
    icd_part = 0.0
    overlap = icd_roots & card_roots
    if overlap:
        icd_part = min(1.0, 0.6 + 0.1 * len(overlap))
    elif icd_list and card_icd:
        for c in icd_list:
            if is_symptom_code(c):
                continue
            for cc in card_icd:
                if c.startswith(_icd_root(cc)) or cc.startswith(_icd_root(c)):
                    icd_part = 0.5
                    break

    pop_mult = _population_match(card, audience)
    if pop_mult == 0:
        risk_flags.append("population_mismatch")
    pop_part = pop_mult

    spec_part = 0.0
    if specialty_slug and card.get("specialty_slug") == specialty_slug:
        spec_part = 1.0
    elif specialty_slug:
        spec_part = 0.2
        risk_flags.append("specialty_weak")

    title_low = (card.get("title") or "").lower()
    path_low = (card.get("source_path") or "").lower()
    blob = title_low + " " + path_low
    hint_score = 0.0
    for hint in hints:
        hint_score += score_card_for_hint(str(hint), blob, icd_list) / 100.0
    hint_score = min(1.0, hint_score)

    diag_part = _diag_text_overlap(diag_text, card)
    exam_blob = " ".join(performed_exams).lower()
    exam_part = 0.3 if exam_blob and any(x in exam_blob for x in title_low.split()[:3]) else 0.0
    compl_part = 0.0
    if complaints:
        cb = " ".join(complaints).lower()
        compl_part = 0.4 if any(w in blob for w in cb.split() if len(w) > 5) else 0.0

    raw = (
        _WEIGHT_ICD * icd_part
        + _WEIGHT_DIAG_TEXT * max(diag_part, hint_score * 0.5)
        + _WEIGHT_SPECIALTY * spec_part
        + _WEIGHT_POPULATION * pop_part
        + _WEIGHT_DEMO * (1.0 if pop_part > 0 else 0.0)
        + _WEIGHT_EXAMS * exam_part
        + _WEIGHT_COMPLAINTS * compl_part
    )
    if (card.get("status") or "active") != "active":
        raw *= 0.7
        risk_flags.append("inactive")

    domain_mult = 1.0
    if _is_venous_icd(icd_list):
        rel = _card_venous_relevance(card)
        if rel >= 0.9:
            raw = min(1.0, raw * 1.15)
        elif rel <= 0.1:
            raw *= 0.12
            risk_flags.append("wrong_nosology_venous")
            domain_mult *= 0.12

    if _is_spine_icd(icd_list):
        rel = _card_spine_bladder_relevance(card, icd_list)
        if rel >= 0.9:
            raw = min(1.0, raw * 1.12)
        elif rel <= 0.05:
            raw *= 0.08
            risk_flags.append("wrong_nosology_spine")
            domain_mult *= 0.08

    clin_mult = clinical_relevance_multiplier(
        card, icd_codes=icd_list, complaints=complaints, ambulatory=True,
    )
    if clin_mult < 0.2:
        risk_flags.append("inpatient_only" if clin_mult < 0.35 else "low_clinical_fit")
    score = round(max(0.0, min(100.0, raw * 100 * clin_mult)), 2)

    icd_fit = icd_fit_for_card(card, icd_list)
    if not icd_fit and icd_list:
        risk_flags.append("low_icd_fit")

    breakdown = {
        "icd": round(icd_part, 2),
        "diagnosis_text": round(diag_part, 2),
        "specialty": round(spec_part, 2),
        "population": round(pop_part, 2),
        "complaints": round(compl_part, 2),
        "exams": round(exam_part, 2),
        "clinical_multiplier": round(clin_mult, 2),
        "domain_multiplier": round(domain_mult, 2),
    }

    title = (card.get("title") or "")[:80]
    icd_s = ", ".join(icd_list[:2])
    reason = f"МКБ {icd_s}, балл {score:.0f}"
    if title:
        reason = f"{reason} — «{title}»"

    return {
        "match_score": score,
        "match_breakdown": breakdown,
        "icd_fit": icd_fit,
        "icd_fit_label": ", ".join(f"{x['code']} ({x['weight']:.2f})" for x in icd_fit[:4]),
        "pick_risk_flags": list(dict.fromkeys(risk_flags)),
        "pick_reason_ru": reason,
        "rejected": score < 22,
    }
