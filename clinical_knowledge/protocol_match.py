"""Подбор карточек протоколов по фактам из КЗ."""
from __future__ import annotations

import re
from typing import Any

from .applicability import assess_card_applicability
from .condition_registry import score_card_for_hint
from .diagnosis_icd import is_symptom_code, prioritize_codes
from .loader import load_protocol_cards_registry

# Веса match_score (ТЗ improve_kz §8.1), нормализуются к ~100
_WEIGHT_ICD = 0.40
_WEIGHT_DIAG_TEXT = 0.20
_WEIGHT_SPECIALTY = 0.15
_WEIGHT_POPULATION = 0.10
_WEIGHT_DEMO = 0.05
_WEIGHT_EXAMS = 0.05
_WEIGHT_COMPLAINTS = 0.05


def _icd_root(code: str) -> str:
    c = (code or "").upper().strip()
    return c[:3] if len(c) >= 3 else c


_VENOUS_ICD_ROOTS = frozenset({"I80", "I81", "I82", "I83", "I84", "I85", "I86", "I87", "I88", "I89"})
_VENOUS_CARD_NEEDLES = (
    "тромбоз", "тгв", "тромбоэмбол", "флеб", "вен ", "веноз", "варикоз", "тромбофлеб",
    "флеботромб", "глубоких вен", "поверхностн",
)
_HEART_FAILURE_NEEDLES = (
    "недостаточност", "сердечн", "кардиомиопат", "функциональн класс", "nyha",
)


def _is_venous_icd(icd_list: list[str]) -> bool:
    for c in icd_list:
        root = _icd_root(c)
        if root in _VENOUS_ICD_ROOTS or (len(root) >= 2 and root.startswith("I8")):
            return True
    return False


def _card_venous_relevance(card: dict[str, Any]) -> float:
    """1.0 – явно венозный КП; 0.0 – явно ЧСН без вен; 0.5 – нейтрально."""
    blob = ((card.get("title") or "") + " " + (card.get("source_path") or "")).lower()
    venous = any(n in blob for n in _VENOUS_CARD_NEEDLES)
    heart = any(n in blob for n in _HEART_FAILURE_NEEDLES)
    if venous and not heart:
        return 1.0
    if heart and not venous:
        return 0.0
    if venous and heart:
        return 0.7
    return 0.5


def _population_match(card_pop: str, consult_audience: str | None) -> float:
    cp = (card_pop or "any").lower()
    ca = (consult_audience or "").lower()
    if not ca or cp == "any":
        return 1.0
    if cp == ca:
        return 1.0
    return 0.0


def _diag_text_overlap(diag_text: str, card: dict[str, Any]) -> float:
    if not diag_text:
        return 0.0
    title = (card.get("title") or "").lower()
    cond = (card.get("condition_label") or "").lower()
    words = [w for w in re.split(r"\W+", diag_text.lower()) if len(w) > 4][:12]
    if not words:
        return 0.0
    blob = title + " " + cond
    hits = sum(1 for w in words if w in blob)
    return min(1.0, hits / max(3, len(words) * 0.4))


def compute_match_score(
    card: dict[str, Any],
    *,
    icd_list: list[str],
    audience: str | None,
    hints: set[str],
    specialty_slug: str | None,
    diag_text: str,
    complaints: list[str],
    performed_exams: list[str],
) -> float:
    """Нормализованный 0–100 match score."""
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

    pop_mult = _population_match(str(card.get("population") or "any"), audience)
    pop_part = pop_mult if pop_mult > 0 else 0.0

    spec_part = 0.0
    if specialty_slug and card.get("specialty_slug") == specialty_slug:
        spec_part = 1.0
    elif specialty_slug:
        spec_part = 0.2

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
    if pop_mult == 0:
        raw *= 0.2
    if (card.get("status") or "active") != "active":
        raw *= 0.7

    if _is_venous_icd(icd_list):
        rel = _card_venous_relevance(card)
        if rel >= 0.9:
            raw = min(1.0, raw * 1.15)
        elif rel <= 0.1:
            raw *= 0.12

    return round(max(0.0, min(100.0, raw * 100)), 2)


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
    icd_list = prioritize_codes([str(x).upper() for x in (cons.get("icd10") or []) if x])
    icd_roots = {_icd_root(c) for c in icd_list}
    audience = ctx.get("adult_or_child")
    hints = set(cons.get("conditions_hint") or [])
    diag_text = str(cons.get("diagnosis_text") or "")
    complaints = list(cons.get("complaints") or [])
    performed = list(cons.get("performed_exams") or [])

    scored: list[tuple[float, dict[str, Any]]] = []
    for card in cards:
        score = compute_match_score(
            card,
            icd_list=icd_list,
            audience=audience,
            hints=hints,
            specialty_slug=specialty_slug,
            diag_text=diag_text,
            complaints=complaints,
            performed_exams=performed,
        )
        if score > 0:
            scored.append((score, card))

    scored.sort(key=lambda x: (-x[0], x[1].get("protocol_id") or ""))
    out: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for sc, card in scored:
        # Дедуп: один протокол (по source_path/protocol_id) — одна строка с лучшим score.
        key = str(card.get("source_path") or card.get("protocol_id") or id(card))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(
            {
                "protocol_id": card.get("protocol_id"),
                "title": card.get("title"),
                "source_path": card.get("source_path"),
                "population": card.get("population"),
                "icd10_primary": card.get("icd10_primary"),
                "match_score": round(sc, 2),
                "approval": card.get("approval"),
                "matched_condition": card.get("condition_label") or card.get("title"),
                "specialty_slug": card.get("specialty_slug"),
            }
        )
        if len(out) >= limit:
            break
    return out


def match_protocol_cards_for_diagnoses(
    consult_facts: dict[str, Any],
    diagnoses: list[dict[str, Any]],
    *,
    specialty_slug: str | None = None,
    limit_per_dx: int = 3,
    limit_total: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Подбор протоколов отдельно по каждому диагнозу.

    Returns (applicable_matches, not_applicable_matches).
    """
    applicable: list[dict[str, Any]] = []
    not_applicable: list[dict[str, Any]] = []
    seen: set[str] = set()

    for dx in diagnoses or [{}]:
        dx_id = str(dx.get("diagnosis_id") or "")
        icd = [dx["icd10_code"]] if dx.get("icd10_code") else []
        facts = dict(consult_facts)
        cons = dict(facts.get("consultation") or {})
        cons["icd10"] = prioritize_codes(list(cons.get("icd10") or []) + icd)
        cons["diagnosis_text"] = dx.get("raw_text") or cons.get("diagnosis_text") or ""
        facts["consultation"] = cons
        if dx.get("certainty") == "suspected":
            cons["conditions_hint"] = list(set(cons.get("conditions_hint") or []) | {"suspected"})

        matches = match_protocol_cards(facts, specialty_slug=specialty_slug, limit=limit_per_dx)
        patient = consult_facts.get("patient_context") or {}
        enriched = annotate_applicability(matches, patient)
        for m in enriched:
            key = str(m.get("source_path") or m.get("protocol_id") or "")
            m["diagnosis_id"] = dx_id
            if m.get("applicability") == "not_applicable":
                if key not in seen:
                    not_applicable.append(m)
                    seen.add(key)
            elif key not in seen and len(applicable) < limit_total:
                applicable.append(m)
                seen.add(key)
    return applicable, not_applicable


def annotate_applicability(
    matches: list[dict[str, Any]],
    patient: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Аддитивно добавляет к каждому матчу applicability/match_reasons/mismatch_reasons.

    Не меняет существующие поля; исходный список не мутируется (возвращается копия).
    """
    out: list[dict[str, Any]] = []
    for m in matches:
        appl, mr, mmr = assess_card_applicability(m, patient)
        enriched = dict(m)
        enriched["applicability"] = appl
        enriched["match_reasons"] = mr
        enriched["mismatch_reasons"] = mmr
        out.append(enriched)
    return out
