"""Извлечение назначений из текста КЗ (B2C)."""
from __future__ import annotations

import re
from typing import Any

_MED_LINE = re.compile(
    r"(?:таб\.?|капс\.?|р-?р\.?|амп\.?|крем|мазь|гель)\s+"
    r"([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{2,}(?:\s+[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]+)?)"
    r"[^.\n]{0,120}",
    re.I,
)
_NAME_STOP = frozenset(
    {
        "таб",
        "таблетке",
        "таблетки",
        "летке",
        "внутрь",
        "наружно",
        "крем",
        "мазь",
        "гель",
        "раза",
        "сутки",
        "недели",
    }
)
_KNOWN_MEDS = (
    "мидокалм",
    "аэртал",
    "дексалгин",
    "пентоксифиллин",
    "тиогамма",
    "ривароксабан",
    "гидроксихлорохин",
    "гидроксихлорохина",
    "тридерм",
    "азитромицин",
    "аспирин",
    "омепразол",
)
_DOSE = re.compile(r"(\d+(?:[.,]\d+)?)\s*мг", re.I)
_FREQ = re.compile(r"(\d+)\s*раз(?:а)?\s*(?:/|в\s*)?(?:сутки|день)", re.I)
_QTY = re.compile(r"№\s*(\d+)", re.I)
_AFTER = re.compile(r"\bпосле\b\s*[:.]?\s*", re.I)
_DURATION = re.compile(r"(\d+)\s*(?:дн|дней|нед|недель|мес)", re.I)


def _valid_name(name: str) -> bool:
    n = (name or "").strip()
    if len(n) < 4:
        return False
    low = n.lower()
    if low in _NAME_STOP:
        return False
    if low.endswith("летке") or low.endswith(" таб"):
        return False
    return True


def _display_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return n
    if n[0].islower():
        return n[0].upper() + n[1:]
    return n


def _clarity_issues(line: str) -> list[str]:
    issues: list[str] = []
    low = line.lower()
    if not _DURATION.search(low) and "постоянно" not in low:
        issues.append("duration_missing")
    if _AFTER.search(low):
        issues.append("ambiguous_start_condition")
        issues.append("unclear_after_what")
    if not _FREQ.search(low) and "постоянно" not in low and "утром" not in low and "раза/сутки" not in low:
        issues.append("frequency_missing")
    return issues


def extract_medications_from_text(text: str) -> list[dict[str, Any]]:
    raw = text or ""
    meds: list[dict[str, Any]] = []
    seen: set[str] = set()

    for m in _MED_LINE.finditer(raw):
        name = (m.group(1) or "").strip()
        if not _valid_name(name):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        line = m.group(0)
        dose_m = _DOSE.search(line)
        freq_m = _FREQ.search(line)
        qty_m = _QTY.search(line)
        dur_m = _DURATION.search(line)
        route = "в/м" if "в/м" in line.lower() else ("наружно" if "наружно" in line.lower() else None)
        meds.append(
            {
                "name": _display_name(name)[:64],
                "dose": (dose_m.group(0) if dose_m else None),
                "quantity": (f"№{qty_m.group(1)}" if qty_m else None),
                "frequency": (freq_m.group(0) if freq_m else None),
                "route": route,
                "duration": dur_m.group(0) if dur_m else None,
                "start_condition": "после" if _AFTER.search(line.lower()) else None,
                "clarity_issues": _clarity_issues(line),
                "source_text": line.strip()[:180],
            }
        )

    low = raw.lower()
    for known in _KNOWN_MEDS:
        if not re.search(rf"\b{re.escape(known)}\b", low):
            continue
        if known in seen:
            continue
        seen.add(known)
        idx = low.find(known)
        ctx = raw[max(0, idx - 8): idx + 90].strip() if idx >= 0 else known
        meds.append(
            {
                "name": _display_name(known),
                "dose": None,
                "quantity": None,
                "frequency": None,
                "route": None,
                "duration": None,
                "clarity_issues": _clarity_issues(ctx),
                "source_text": ctx[:180],
            }
        )

    return meds[:20]


def medications_patient_summary(meds: list[dict[str, Any]]) -> str:
    if not meds:
        return ""
    names = [str(m.get("name") or "") for m in meds if m.get("name")]
    if not names:
        return ""
    has_after = any(m.get("start_condition") == "после" for m in meds)
    has_duration_gap = any("duration_missing" in (m.get("clarity_issues") or []) for m in meds)
    base = f"Лечение назначено ({len(names)} препарат(ов): {', '.join(names[:5])}"
    if len(names) > 5:
        base += f" и ещё {len(names) - 5}"
    base += ")."
    notes: list[str] = []
    if has_duration_gap:
        notes.append("Стоит уточнить длительность приёма.")
    if has_after:
        notes.append("Уточните, что означает этап «после» в схеме лечения.")
    if notes:
        base += " " + " ".join(notes)
    return base
