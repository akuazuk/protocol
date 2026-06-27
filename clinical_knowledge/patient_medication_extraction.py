"""Извлечение назначений из текста КЗ (B2C)."""
from __future__ import annotations

import re
from typing import Any

_MED_LINE = re.compile(
    r"(?:таб\.?|капс\.?|р-?р\.?|амп\.?)\s*"
    r"([A-Za-zА-Яа-яЁё\-]+(?:\s+[A-Za-zА-Яа-яЁё\-]+)?)"
    r"[^.\n]{0,120}",
    re.I,
)
_KNOWN_MEDS = (
    "мидокалм",
    "аэртал",
    "дексалгин",
    "пентоксифиллин",
    "тиогамма",
    "ривароксабан",
    "аспирин",
    "омепразол",
)
_DOSE = re.compile(r"(\d+(?:[.,]\d+)?)\s*мг", re.I)
_FREQ = re.compile(r"(\d+)\s*раз(?:а)?\s*в\s*день", re.I)
_QTY = re.compile(r"№\s*(\d+)", re.I)
_AFTER = re.compile(r"\bпосле\b\s*[:.]?\s*", re.I)


def _clarity_issues(line: str) -> list[str]:
    issues: list[str] = []
    low = line.lower()
    if not re.search(r"\d+\s*(?:дн|нед|мес)", low) and "постоянно" not in low:
        issues.append("duration_missing")
    if _AFTER.search(low):
        issues.append("ambiguous_start_condition")
        issues.append("unclear_after_what")
    if not _FREQ.search(low) and "постоянно" not in low and "утром" not in low:
        issues.append("frequency_missing")
    return issues


def extract_medications_from_text(text: str) -> list[dict[str, Any]]:
    raw = text or ""
    meds: list[dict[str, Any]] = []
    seen: set[str] = set()

    for m in _MED_LINE.finditer(raw):
        name = (m.group(1) or "").strip()
        if len(name) < 3:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        line = m.group(0)
        dose_m = _DOSE.search(line)
        freq_m = _FREQ.search(line)
        qty_m = _QTY.search(line)
        route = "в/м" if "в/м" in line.lower() else None
        meds.append(
            {
                "name": name[:64],
                "dose": (dose_m.group(0) if dose_m else None),
                "quantity": (f"№{qty_m.group(1)}" if qty_m else None),
                "frequency": (freq_m.group(0) if freq_m else None),
                "route": route,
                "duration": None,
                "start_condition": "после" if _AFTER.search(line.lower()) else None,
                "clarity_issues": _clarity_issues(line),
                "source_text": line.strip()[:180],
            }
        )

    low = raw.lower()
    for known in _KNOWN_MEDS:
        if known in low and known not in seen:
            seen.add(known)
            ctx = ""
            idx = low.find(known)
            if idx >= 0:
                ctx = raw[max(0, idx - 5): idx + 80].strip()
            meds.append(
                {
                    "name": known.capitalize() if known.isascii() else known.title(),
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
    has_clarity = any(m.get("clarity_issues") for m in meds)
    base = f"Лечение назначено ({len(names)} препарат(ов): {', '.join(names[:5])}"
    if len(names) > 5:
        base += f" и ещё {len(names) - 5}"
    base += ")."
    if has_clarity:
        base += " Стоит уточнить длительность приёма и что означает этап «после» в схеме лечения."
    return base
