"""Verify LLM quotes against source text."""
from __future__ import annotations

import re
from typing import Any


def _norm(s: str) -> str:
    t = (s or "").lower().replace("\u00ad", "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def quote_found_in_source(quote: str, source_blob: str, *, min_ratio: float = 0.55) -> bool:
    q = _norm(quote)
    if len(q) < 12:
        return True
    blob = _norm(source_blob)
    if q in blob:
        return True
    words = [w for w in re.findall(r"[а-яёa-z0-9]{4,}", q) if len(w) >= 4]
    if not words:
        return True
    hits = sum(1 for w in words[:8] if w in blob)
    return hits / max(len(words[:8]), 1) >= min_ratio


def validate_quotes_in_payload(payload: dict[str, Any], source_blob: str) -> list[str]:
    issues: list[str] = []

    def check_item(item: dict, path: str) -> None:
        q = str(item.get("quote") or "")
        if q and not quote_found_in_source(q, source_blob):
            issues.append(f"quote_not_in_source:{path}")

    for i, ex in enumerate(payload.get("required_exams") or []):
        if isinstance(ex, dict):
            check_item(ex, f"exams[{i}]")
    for i, d in enumerate(payload.get("drugs") or []):
        if isinstance(d, dict):
            check_item(d, f"drugs[{i}]")
    for i, rf in enumerate(payload.get("red_flags") or []):
        if isinstance(rf, dict):
            check_item(rf, f"red_flags[{i}]")
    return issues
