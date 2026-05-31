"""Оценка применимости протокола к пациенту (ТЗ раздел 12).

Ключевые правила:
- детский протокол не применяется к взрослому автоматически (и наоборот);
- протокол «только для беременных» применим лишь при признаках беременности;
- при неизвестных данных пациента — possibly_applicable/unknown, не not_applicable.
"""
from __future__ import annotations

from typing import Any

PREGNANCY_MARKERS = ("беременн", "гестац", "беременности", "родоразрешен", "послеродов")
NEONATAL_MARKERS = ("новорожд", "перинатал", "неонат")


def _card_text(card: dict[str, Any]) -> str:
    return ((card.get("title") or "") + " " + (card.get("source_path") or "")).lower()


def assess_card_applicability(
    card: dict[str, Any],
    patient: dict[str, Any] | None,
) -> tuple[str, list[str], list[str]]:
    """Возвращает (applicability, match_reasons, mismatch_reasons).

    applicability ∈ {applicable, possibly_applicable, not_applicable, unknown}.
    ``patient``: {age_years, adult_or_child, sex, pregnancy}.
    """
    p = patient or {}
    audience = (p.get("adult_or_child") or "").lower() or None
    age_years = p.get("age_years")
    sex = (p.get("sex") or "").lower() or None
    pregnancy = p.get("pregnancy")

    card_pop = str(card.get("population") or "any").lower()
    text = _card_text(card)
    match_reasons: list[str] = []
    mismatch_reasons: list[str] = []

    # --- популяция (возраст) ---
    pop_state = "ok"
    if card_pop in ("child", "children", "pediatric"):
        if audience == "adult":
            pop_state = "block"
            mismatch_reasons.append("Протокол детский, пациент взрослый.")
        elif audience in ("child", "newborn"):
            match_reasons.append("Возрастная группа совпадает (детский протокол).")
        else:
            pop_state = "soft"
    elif card_pop == "adult":
        if audience in ("child", "newborn"):
            pop_state = "block"
            mismatch_reasons.append("Протокол для взрослых, пациент ребёнок.")
        elif audience == "adult":
            match_reasons.append("Возрастная группа совпадает (взрослый протокол).")
        else:
            pop_state = "soft"
    else:  # any / unknown
        if audience:
            match_reasons.append("Протокол без возрастных ограничений.")

    # --- беременность ---
    preg_state = "ok"
    is_pregnancy_protocol = any(mk in text for mk in PREGNANCY_MARKERS)
    if is_pregnancy_protocol:
        if pregnancy is True:
            match_reasons.append("Протокол для беременных, у пациентки указана беременность.")
        elif sex == "male":
            preg_state = "block"
            mismatch_reasons.append("Протокол для беременных неприменим к пациенту мужского пола.")
        elif pregnancy is None:
            preg_state = "soft"
            mismatch_reasons.append("Протокол для беременных: беременность в КЗ не подтверждена.")
        else:  # pregnancy is False
            preg_state = "block"
            mismatch_reasons.append("Протокол для беременных, признаков беременности нет.")

    if any(mk in text for mk in NEONATAL_MARKERS) and audience == "adult":
        pop_state = "block"
        mismatch_reasons.append("Протокол неонатальный/перинатальный, пациент взрослый.")

    # --- агрегирование ---
    if pop_state == "block" or preg_state == "block":
        return "not_applicable", match_reasons, mismatch_reasons
    if pop_state == "soft" or preg_state == "soft":
        return "possibly_applicable", match_reasons, mismatch_reasons
    if audience is None and age_years is None and not match_reasons:
        return "unknown", match_reasons, mismatch_reasons
    return "applicable", match_reasons, mismatch_reasons
