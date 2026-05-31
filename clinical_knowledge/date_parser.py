"""Разбор дат из консультативных заключений (stdlib-only).

Поддерживает форматы ДД.ММ.ГГ, ДД.ММ.ГГГГ, ДД/ММ/ГГГГ, ДД-ММ-ГГГГ
и словесные даты «14 июля 2024».
"""
from __future__ import annotations

import datetime as _dt
import re

_MONTHS_RU = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "май": 5,
    "июн": 6, "июл": 7, "август": 8, "сентябр": 9, "октябр": 10,
    "ноябр": 11, "декабр": 12,
}

RE_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})\b")
RE_WORD_DATE = re.compile(
    r"\b(\d{1,2})\s+([а-яё]{3,})\.?\s+(\d{4})\b",
    re.I,
)
RE_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def _normalize_year(y: int) -> int:
    if y < 100:
        # 24 -> 2024; пороговое значение 70 для эпохи 1900/2000
        return 2000 + y if y <= 69 else 1900 + y
    return y


def parse_date(text: str | None) -> _dt.date | None:
    """Первая распознанная дата из строки/фрагмента."""
    if not text:
        return None
    m = RE_NUMERIC_DATE.search(text)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), _normalize_year(int(m.group(3)))
        try:
            return _dt.date(y, mo, d)
        except ValueError:
            pass
    m = RE_WORD_DATE.search(text)
    if m:
        d = int(m.group(1))
        mon_raw = m.group(2).lower()
        y = int(m.group(3))
        mo = None
        for stem, num in _MONTHS_RU.items():
            if mon_raw.startswith(stem):
                mo = num
                break
        if mo:
            try:
                return _dt.date(y, mo, d)
            except ValueError:
                pass
    return None


def parse_all_dates(text: str | None) -> list[_dt.date]:
    """Все распознанные даты (числовые и словесные), без дублей, в порядке появления."""
    out: list[_dt.date] = []
    if not text:
        return out
    seen: set[_dt.date] = set()
    spans: list[tuple[int, _dt.date]] = []
    for m in RE_NUMERIC_DATE.finditer(text):
        d, mo, y = int(m.group(1)), int(m.group(2)), _normalize_year(int(m.group(3)))
        try:
            dt = _dt.date(y, mo, d)
        except ValueError:
            continue
        spans.append((m.start(), dt))
    for m in RE_WORD_DATE.finditer(text):
        d = int(m.group(1))
        mon_raw = m.group(2).lower()
        y = int(m.group(3))
        mo = next((num for stem, num in _MONTHS_RU.items() if mon_raw.startswith(stem)), None)
        if not mo:
            continue
        try:
            dt = _dt.date(y, mo, d)
        except ValueError:
            continue
        spans.append((m.start(), dt))
    for _, dt in sorted(spans, key=lambda x: x[0]):
        if dt not in seen:
            seen.add(dt)
            out.append(dt)
    return out


def parse_time(text: str | None) -> _dt.time | None:
    if not text:
        return None
    m = RE_TIME.search(text)
    if not m:
        return None
    try:
        return _dt.time(int(m.group(1)), int(m.group(2)))
    except ValueError:
        return None
