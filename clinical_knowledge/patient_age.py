"""Возраст пациента на дату визита: готовые годы или расчёт от ДР.

Не логировать дату рождения.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

_ISO = re.compile(r"^(\d{4})[-./](\d{1,2})[-./](\d{1,2})")
_DMY = re.compile(r"^(\d{1,2})[-./](\d{1,2})[-./](\d{4})")
_AGE_KEYS = ("patient_age_years", "age_years", "age", "patient_age")
_BDATE_KEYS = ("patient_bdate", "birth_date", "bdate", "patient_birth_date")
_VISIT_KEYS = ("visit_date", "date", "visit_date_iso_db", "visit_date_iso")


def parse_iso_date(raw: Any) -> date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text[:10]
    match = _ISO.match(text)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    else:
        match = _DMY.match(text)
        if not match:
            return None
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
    try:
        return date(year, month, day)
    except ValueError:
        return None


def age_years_on(born: date, on: date) -> int | None:
    if on < born:
        return None
    years = on.year - born.year
    if (on.month, on.day) < (born.month, born.day):
        years -= 1
    if years < 0 or years > 120:
        return None
    return years


def _first_number(sources: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    for src in sources:
        for key in keys:
            raw = src.get(key)
            if raw is None or raw == "":
                continue
            try:
                value = float(str(raw).replace(",", ".").strip())
            except ValueError:
                continue
            if 0 <= value <= 120:
                return value
    return None


def _first_date(sources: list[dict[str, Any]], keys: tuple[str, ...]) -> date | None:
    for src in sources:
        for key in keys:
            parsed = parse_iso_date(src.get(key))
            if parsed:
                return parsed
    return None


def resolve_patient_age(
    clinical: dict[str, Any] | None = None,
    record: dict[str, Any] | None = None,
    *,
    today: date | None = None,
) -> dict[str, Any]:
    """Возраст: готовые годы, иначе ДР + дата визита (не сегодня, кроме fallback визита)."""
    sources = [src for src in (clinical, record) if isinstance(src, dict)]
    years = _first_number(sources, _AGE_KEYS)
    visit = _first_date(sources, _VISIT_KEYS)
    born = _first_date(sources, _BDATE_KEYS)
    source = "none"
    if years is not None:
        source = "age_years"
        age = int(years)
    elif born is not None:
        on = visit or today or date.today()
        computed = age_years_on(born, on)
        if computed is None:
            age = None
        else:
            age = computed
            source = "bdate_visit" if visit else "bdate_today"
    else:
        age = None
    if age is None:
        audience = "unknown"
    else:
        audience = "child" if age < 18 else "adult"
    return {
        "age_years": age,
        "audience": audience,
        "visit_date": visit.isoformat() if visit else None,
        "age_source": source,
    }
