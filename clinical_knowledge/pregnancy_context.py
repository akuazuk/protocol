"""Детекция текущей беременности vs анамнез «беременности - N»."""
from __future__ import annotations

import re


def is_active_pregnancy(
    raw: str,
    icd: list[str] | None = None,
    *,
    age_years: int | float | None = None,
) -> bool:
    """True только при текущей беременности (O*, срок), не при «беременности - 1» в анамнезе."""
    icd_up = [str(c).upper() for c in (icd or []) if c]
    if any(c.startswith("O") for c in icd_up):
        return True
    if isinstance(age_years, (int, float)) and age_years > 50:
        return False

    low = (raw or "").lower()
    if not re.search(r"беремен|гesta", low):
        return False

    current_markers = (
        r"беременност[ьи]\s+\d+\s*(?:нед|мес|триместр)",
        r"срок\s+беремен",
        r"беременна\b",
        r"на\s+уч[её]те\s+.*беремен",
        r"\bo\d{2}\.\d",
    )
    if any(re.search(p, low) for p in current_markers):
        return True

    # Анамнез: «беременности - 1, роды-1» - не текущая беременность
    if re.search(r"беременност[ьи]\s*[-–:]\s*\d", low):
        return False
    if re.search(r"беременност[ьи]\s+в\s+анамнез", low):
        return False

    return bool(re.search(r"\bбеременност[ьи]\b", low) and "гестаци" in low)
