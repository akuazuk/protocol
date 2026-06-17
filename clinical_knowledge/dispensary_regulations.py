"""Регламенты наблюдения (НПА Минздрава) для оценки блока «Контроль» в КЗ."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REG_DIR = ROOT / "data" / "regulations"


@lru_cache(maxsize=4)
def _load_regulation(name: str) -> dict[str, Any]:
    path = REG_DIR / f"{name}.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def icd_chapter(code: str) -> str:
    c = (code or "").strip().upper()
    return c[0] if c and c[0].isalpha() else ""


def lookup_follow_up_expectations(icd_codes: list[str]) -> dict[str, Any]:
    """Ожидания по наблюдению: НПА + глава МКБ."""
    reg = _load_regulation("mz_2015_127")
    by_chapter = reg.get("follow_up_by_icd_chapter") or {}
    default = reg.get("default_follow_up") or {}
    chapters: list[str] = []
    hints: list[str] = []
    min_months: int | None = None

    for code in icd_codes or []:
        ch = icd_chapter(code)
        if ch and ch not in chapters:
            chapters.append(ch)
        entry = by_chapter.get(ch) if ch else None
        if isinstance(entry, dict):
            hint = (entry.get("hint") or "").strip()
            if hint and hint not in hints:
                hints.append(hint)
            m = entry.get("min_interval_months")
            if isinstance(m, int):
                min_months = m if min_months is None else min(min_months, m)

    if not hints:
        dh = (default.get("hint") or "").strip()
        if dh:
            hints.append(dh)
        m = default.get("min_interval_months")
        if isinstance(m, int):
            min_months = m

    conclusion_sec = None
    for sec in reg.get("sections") or []:
        if sec.get("id") == "conclusion_requirements":
            conclusion_sec = sec
            break

    return {
        "regulation_id": reg.get("id") or "mz_2015_127",
        "regulation_title": reg.get("title") or "",
        "regulation_source": reg.get("source") or "",
        "regulation_url": reg.get("url") or "",
        "icd_chapters": chapters,
        "follow_up_hints": hints,
        "min_interval_months": min_months,
        "conclusion_requirement": (conclusion_sec or {}).get("follow_up_hint") or "",
        "exam_structure_requirement": next(
            (
                (s.get("text") or "")[:400]
                for s in (reg.get("sections") or [])
                if s.get("id") == "exam_structure"
            ),
            "",
        ),
    }


def completeness_requirements_from_regulation() -> dict[str, str]:
    """Требования к полноте секций КЗ из НПА (не из КП)."""
    reg = _load_regulation("mz_2015_127")
    out: dict[str, str] = {}
    for sec in reg.get("sections") or []:
        if sec.get("id") != "exam_structure":
            continue
        text = (sec.get("text") or "").strip()
        for key in sec.get("completeness_sections") or []:
            out[str(key)] = text
    return out


def follow_up_mentioned_in_text(text: str, *, min_months: int | None = None) -> bool:
    low = (text or "").lower()
    if not low:
        return False
    markers = (
        "контрольн", "повторн", "явк", "осмотр через", "диспансер",
        "наблюден", "через месяц", "через недел", "через год",
    )
    if any(m in low for m in markers):
        return True
    if min_months and re.search(rf"через\s+{min_months}\s+месяц", low):
        return True
    return bool(re.search(r"через\s+\d+\s+месяц", low))
