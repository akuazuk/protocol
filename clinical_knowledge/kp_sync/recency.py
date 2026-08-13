"""Вес свежести постановления: новый КП не должен проигрывать старому при том же МКБ."""
from __future__ import annotations

from datetime import date
from typing import Any


def approval_year(card: dict[str, Any]) -> int | None:
    appr = card.get("approval") if isinstance(card.get("approval"), dict) else {}
    raw = (
        str(appr.get("date") or "")
        or str(card.get("approval_date") or "")
        or ""
    )
    raw = raw.strip()
    if len(raw) >= 4 and raw[:4].isdigit():
        year = int(raw[:4])
        if 1990 <= year <= 2100:
            return year
    # ДД.ММ.ГГГГ
    if len(raw) >= 10 and raw[6:10].isdigit():
        year = int(raw[6:10])
        if 1990 <= year <= 2100:
            return year
    return None


def recency_multiplier(card: dict[str, Any], *, today: date | None = None) -> float:
    """1.0 нейтрально. Слабый boost свежим, лёгкий штраф очень старым.

    Не перебивает ICD-first: применять только после icd_part, множитель узкий.
    """
    if str(card.get("status") or "active").lower() in {"superseded", "outdated"}:
        return 0.55
    year = approval_year(card)
    if year is None:
        return 1.0
    now_y = (today or date.today()).year
    age = max(0, now_y - year)
    if age <= 0:
        return 1.12
    if age == 1:
        return 1.08
    if age <= 3:
        return 1.04
    if age >= 12:
        return 0.88
    if age >= 8:
        return 0.94
    return 1.0
