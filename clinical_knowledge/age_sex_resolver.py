"""Возраст/пол пациента и возрастные группы (stdlib-only).

Возраст рассчитывается на дату консультации (ТЗ раздел 9). Если дата консультации
не распознана, используется текущая дата как fallback с выставлением warning.
"""
from __future__ import annotations

import datetime as _dt
import re

RE_SEX_F = re.compile(r"\b(женский|жен\.?\s*пол|пол\s*[:\-]?\s*ж(?:ен)?)\b", re.I)
RE_SEX_M = re.compile(r"\b(мужской|муж\.?\s*пол|пол\s*[:\-]?\s*м(?:уж)?)\b", re.I)
RE_PREG = re.compile(r"\b(беременн|гестац|беременность\s+\d+)\w*", re.I)

# «Возраст: 48 лет», «48 л.», «3 года», «6 мес»
RE_AGE_YEARS = re.compile(r"\bвозраст\s*[:\-]?\s*(\d{1,3})\s*(?:лет|года?|л\.)\b", re.I)
RE_AGE_INLINE_YEARS = re.compile(r"\b(\d{1,3})\s*(?:лет|года?)\b", re.I)
RE_AGE_MONTHS = re.compile(r"\b(\d{1,2})\s*(?:мес(?:яц)?[а-я]*)\b", re.I)
RE_DOB = re.compile(
    r"(?:дата\s+рождения|д\.?\s*р\.?|г\.?\s*р\.?|род(?:илс[яа]|\.)?)\s*[:\-]?\s*"
    r"(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})",
    re.I,
)


def detect_sex(text: str) -> str:
    if RE_SEX_F.search(text or ""):
        return "female"
    if RE_SEX_M.search(text or ""):
        return "male"
    return "unknown"


def detect_pregnancy(text: str) -> bool:
    return bool(RE_PREG.search(text or ""))


def age_at(birth_date: _dt.date, on_date: _dt.date) -> tuple[int, int]:
    """Возраст в полных годах и месяцах на дату ``on_date``."""
    years = on_date.year - birth_date.year - (
        (on_date.month, on_date.day) < (birth_date.month, birth_date.day)
    )
    months = (on_date.year - birth_date.year) * 12 + (on_date.month - birth_date.month)
    if on_date.day < birth_date.day:
        months -= 1
    return max(years, 0), max(months, 0)


def age_group(age_years: int | None, age_months: int | None = None) -> str:
    """Возрастная группа по ТЗ: newborn/infant/child/adult/elderly/unknown."""
    if age_years is None and age_months is None:
        return "unknown"
    total_months = age_months if age_months is not None else (
        (age_years or 0) * 12
    )
    if age_years is not None and age_years >= 65:
        return "elderly"
    if age_years is not None and age_years >= 18:
        return "adult"
    # < 18 лет — уточняем младенцев/новорождённых по месяцам, если есть
    if age_months is not None:
        if age_months <= 0:
            return "newborn"  # упрощение: <1 мес считаем как newborn-зону
        if age_months < 12:
            return "infant"
    if age_years is not None and age_years < 18:
        return "child"
    if total_months < 1:
        return "newborn"
    if total_months < 12:
        return "infant"
    return "child"


def adult_or_child(age_years: int | None, age_group_value: str | None = None) -> str:
    if age_group_value == "newborn":
        return "newborn"
    if age_years is None:
        return "unknown"
    return "adult" if age_years >= 18 else "child"


def resolve_age(
    text: str,
    *,
    birth_date: _dt.date | None = None,
    consultation_date: _dt.date | None = None,
) -> dict:
    """Возвращает {age_years, age_months, age_group, adult_or_child, warnings, used_fallback}.

    Приоритет: расчёт из ДР на дату консультации → явный «возраст N лет» в тексте.
    """
    warnings: list[str] = []
    used_fallback = False
    age_years: int | None = None
    age_months: int | None = None

    if birth_date is not None:
        on_date = consultation_date
        if on_date is None:
            on_date = _dt.date.today()
            used_fallback = True
            warnings.append(
                "Дата консультации не распознана: возраст рассчитан на текущую дату."
            )
        age_years, age_months = age_at(birth_date, on_date)
    else:
        m = RE_AGE_YEARS.search(text or "") or RE_AGE_INLINE_YEARS.search(text or "")
        if m:
            try:
                age_years = int(m.group(1))
            except (TypeError, ValueError):
                age_years = None
        mm = RE_AGE_MONTHS.search(text or "")
        if mm and age_years is None:
            try:
                age_months = int(mm.group(1))
            except (TypeError, ValueError):
                age_months = None

    grp = age_group(age_years, age_months)
    return {
        "age_years": age_years,
        "age_months": age_months,
        "age_group": grp,
        "adult_or_child": adult_or_child(age_years, grp),
        "warnings": warnings,
        "used_fallback": used_fallback,
    }


def parse_birth_date(text: str) -> _dt.date | None:
    from .date_parser import parse_date

    m = RE_DOB.search(text or "")
    if m:
        return parse_date(m.group(1))
    return None
