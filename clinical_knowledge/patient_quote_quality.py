"""Фильтр цитат протокола для B2C - без мусора и B2B-синтетики."""
from __future__ import annotations

import re
from typing import Any

from .protocol_audience import is_synthetic_summary_excerpt

_FORBIDDEN_SUBSTRINGS = (
    "pmp22",
    "smn1",
    "smn2",
    "прилагается",
    "инсульт у дет",
    "у детей",
    "протокол: ",
    "нозология:",
)
_FORBIDDEN_RE = re.compile(
    r"(?:\bPMP22\b|\bSMN1\b|\bSMN2\b|прилагается\s+к\s+протоколу|"
    r"инсульт\s+у\s+дет|детск(?:ое|ий)\s+население.*инсульт)",
    re.I,
)


def is_unsafe_quote(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 12:
        return True
    if is_synthetic_summary_excerpt(t):
        return True
    low = t.lower()
    if any(s in low for s in _FORBIDDEN_SUBSTRINGS):
        return True
    if _FORBIDDEN_RE.search(t):
        return True
    if low.count(":") >= 4 and "рубрика" in low:
        return True
    return False


def filter_protocol_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in citations:
        if not isinstance(row, dict):
            continue
        excerpt = str(row.get("excerpt") or "")
        if is_unsafe_quote(excerpt):
            continue
        out.append(row)
    return out


def filter_card_excerpts(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Убрать protocol_excerpt из alignment cards если мусор."""
    out: list[dict[str, Any]] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        c = dict(card)
        ex = str(c.get("protocol_excerpt") or "")
        if ex and is_unsafe_quote(ex):
            c["protocol_excerpt"] = ""
        out.append(c)
    return out


def scrub_forbidden_from_patient_report(report: dict[str, Any]) -> dict[str, Any]:
    """Убрать педиатрический/генетический мусор из всех patient-facing полей."""
    if not isinstance(report, dict):
        return report
    out = dict(report)
    for key in ("plain_summary_ru", "headline_ru", "exams_summary_ru", "medications_summary_ru"):
        val = out.get(key)
        if isinstance(val, str) and is_unsafe_quote(val):
            out[key] = sanitize_patient_text(val.split(".")[0] + ".") if "." in val else ""
    blocks = []
    for b in out.get("blocks") or []:
        if not isinstance(b, dict):
            continue
        row = dict(b)
        for key in ("summary_ru", "why_ru"):
            val = str(row.get(key) or "")
            if val and is_unsafe_quote(val):
                row[key] = "Раздел требует уточнения у врача на приёме."
        gaps = []
        for g in row.get("gaps") or []:
            txt = str(g if isinstance(g, str) else g.get("text_ru") or g.get("text") or "")
            if txt and not is_unsafe_quote(txt):
                gaps.append(g)
        if gaps != row.get("gaps"):
            row["gaps"] = gaps
        blocks.append(row)
    out["blocks"] = blocks
    out["protocol_citations"] = filter_protocol_citations(list(out.get("protocol_citations") or []))
    clarify = []
    for item in out.get("clarification_points") or []:
        txt = str(item if isinstance(item, str) else item.get("text_ru") or item.get("text") or "")
        if txt and not is_unsafe_quote(txt):
            clarify.append(item)
    out["clarification_points"] = clarify
    return out


def sanitize_patient_text(text: str) -> str:
    """Убрать запрещённые формулировки из patient-facing текста."""
    t = (text or "").strip()
    replacements = (
        (re.compile(r"по\s+протоколу\s+положено", re.I), "по стандарту лечения обычно указывают"),
        (re.compile(r"по\s+соп\s+", re.I), "по стандарту "),
        (re.compile(r"gate_score", re.I), ""),
        (re.compile(r"send_gate", re.I), ""),
    )
    for pat, repl in replacements:
        t = pat.sub(repl, t)
    return re.sub(r"\s+", " ", t).strip()
