"""Извлечение обследований из текста КЗ (B2C)."""
from __future__ import annotations

import re
from typing import Any

EXAM_SYNONYMS: dict[str, list[str]] = {
    "MRI": ["мрт", "магнитно-резонансная томография", "магнитно-резонансн"],
    "CT": ["кт", "компьютерная томография"],
    "XRAY": ["рентген", "рентгенография", "рентгенограмм"],
    "US": ["узи", "ультразвуковое исследование", "ультразвук"],
    "ECG": ["экг", "электрокардиограмма"],
    "EEG": ["ээг", "электроэнцефалография"],
    "LAB": ["анализ", "оак", "оам", "биохимия"],
}


def _normalize_exam_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def extract_exams_from_text(text: str) -> list[dict[str, Any]]:
    """Найти назначенные обследования в КЗ."""
    raw = text or ""
    low = raw.lower()
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    mri_pat = re.compile(
        r"мрт[^.\n]{0,120}(?:шейн|головн|позвоноч|мозг|гм|шоп|пояснич)",
        re.I,
    )
    for m in mri_pat.finditer(raw):
        snippet = _normalize_exam_line(m.group(0))
        key = snippet.lower()[:40]
        if key in seen:
            continue
        seen.add(key)
        areas: list[str] = []
        if "шейн" in snippet.lower():
            areas.append("шейный отдел позвоночника")
        if "головн" in snippet.lower() or "мозг" in snippet.lower() or " гм" in snippet.lower():
            areas.append("головной мозг")
        found.append(
            {
                "exam_type": "MRI",
                "label_ru": snippet[:160],
                "body_area_ru": areas,
                "deadline": None,
                "status": "recommended",
                "source_text": snippet[:200],
            }
        )

    if not found and "мрт" in low:
        found.append(
            {
                "exam_type": "MRI",
                "label_ru": "МРТ",
                "body_area_ru": [],
                "deadline": None,
                "status": "recommended",
                "source_text": "МРТ",
            }
        )

    for kind, syns in EXAM_SYNONYMS.items():
        if kind == "MRI":
            continue
        for syn in syns:
            if syn in low and kind.lower() not in seen:
                seen.add(kind.lower())
                found.append(
                    {
                        "exam_type": kind,
                        "label_ru": syn.upper() if len(syn) <= 4 else syn.capitalize(),
                        "body_area_ru": [],
                        "deadline": None,
                        "status": "recommended",
                        "source_text": syn,
                    }
                )
                break

    return found[:12]


def exams_patient_summary(exams: list[dict[str, Any]]) -> str:
    if not exams:
        return ""
    labels = [_normalize_exam_line(str(e.get("label_ru") or e.get("source_text") or "")) for e in exams]
    labels = [l for l in labels if l]
    if not labels:
        return ""
    if len(labels) == 1:
        return f"Обследование назначено: {labels[0]}. Стоит уточнить срок выполнения."
    return f"Обследования назначены: {', '.join(labels[:3])}. Стоит уточнить сроки."
