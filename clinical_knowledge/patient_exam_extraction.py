"""Извлечение обследований из текста КЗ (B2C)."""
from __future__ import annotations

import re
from typing import Any

# Короткие аббревиатуры - только отдельные слова (не «кт» в креатинина / «узи» в трансфузии).
_IMAGING_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bмрт\b", re.I), "MRI", "МРТ"),
    (re.compile(r"\b(?:компьютерная\s+)?кт\b", re.I), "CT", "КТ"),
    (re.compile(r"\bузи\b", re.I), "US", "УЗИ"),
    (re.compile(r"\b(?:рентген(?:ография)?|рентгенограмм\w*)\b", re.I), "XRAY", "Рентген"),
    (re.compile(r"\bэкг\b", re.I), "ECG", "ЭКГ"),
    (re.compile(r"\bээг\b", re.I), "EEG", "ЭЭГ"),
)

_LAB_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bоак\b|\bобщ(?:ий|его)\s+анализ\s+крови\b", re.I), "LAB_OAK", "Общий анализ крови (ОАК)"),
    (re.compile(r"\bоам\b|\bанализ\s+мочи\b|\bобщ(?:ий|его)\s+анализ\s+мочи\b", re.I), "LAB_OAM", "Общий анализ мочи (ОАМ)"),
    (re.compile(r"\bбиохимич\w*\s+анализ\b|\bанализ\s+крови\s+биохим\b", re.I), "LAB_BIO", "Биохимический анализ крови"),
)

_MRI_DETAIL = re.compile(
    r"мрт[^.\n]{0,120}(?:шейн|головн|позвоноч|мозг|гм|шоп|пояснич)",
    re.I,
)


def _normalize_exam_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def extract_exams_from_text(text: str) -> list[dict[str, Any]]:
    """Найти назначенные обследования и анализы в КЗ."""
    raw = text or ""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    for m in _MRI_DETAIL.finditer(raw):
        snippet = _normalize_exam_line(m.group(0))
        key = f"mri:{snippet.lower()[:40]}"
        if key in seen:
            continue
        seen.add(key)
        areas: list[str] = []
        sl = snippet.lower()
        if "шейн" in sl:
            areas.append("шейный отдел позвоночника")
        if "головн" in sl or "мозг" in sl:
            areas.append("головной мозг")
        found.append(
            {
                "exam_type": "MRI",
                "category": "imaging",
                "label_ru": snippet[:160],
                "body_area_ru": areas,
                "deadline": None,
                "status": "recommended",
                "source_text": snippet[:200],
            }
        )

    if not any(e.get("exam_type") == "MRI" for e in found) and re.search(r"\bмрт\b", raw, re.I):
        found.append(
            {
                "exam_type": "MRI",
                "category": "imaging",
                "label_ru": "МРТ",
                "body_area_ru": [],
                "deadline": None,
                "status": "recommended",
                "source_text": "МРТ",
            }
        )

    for pat, kind, label in _IMAGING_PATTERNS:
        if kind == "MRI" and any(e.get("exam_type") == "MRI" for e in found):
            continue
        if pat.search(raw) and kind not in seen:
            seen.add(kind)
            found.append(
                {
                    "exam_type": kind,
                    "category": "imaging",
                    "label_ru": label,
                    "body_area_ru": [],
                    "deadline": None,
                    "status": "recommended",
                    "source_text": label,
                }
            )

    for pat, kind, label in _LAB_PATTERNS:
        if kind not in seen and pat.search(raw):
            seen.add(kind)
            found.append(
                {
                    "exam_type": kind,
                    "category": "lab",
                    "label_ru": label,
                    "body_area_ru": [],
                    "deadline": None,
                    "status": "recommended",
                    "source_text": label,
                }
            )

    return found[:12]


def imaging_exams(exams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in exams if e.get("category") == "imaging" or str(e.get("exam_type", "")).startswith(("MRI", "CT", "US", "XRAY", "ECG", "EEG"))]


def lab_exams(exams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in exams if e.get("category") == "lab" or str(e.get("exam_type", "")).startswith("LAB_")]


def exams_patient_summary(exams: list[dict[str, Any]]) -> str:
    if not exams:
        return ""
    imaging = imaging_exams(exams)
    labs = lab_exams(exams)
    parts: list[str] = []
    if imaging:
        labels = [_normalize_exam_line(str(e.get("label_ru") or "")) for e in imaging]
        labels = [l for l in labels if l]
        if labels:
            parts.append(
                f"Инструментальные обследования: {', '.join(labels[:3])}"
                + (f" и ещё {len(labels) - 3}" if len(labels) > 3 else "")
            )
    if labs:
        labels = [_normalize_exam_line(str(e.get("label_ru") or "")) for e in labs]
        labels = [l for l in labels if l]
        if labels:
            parts.append(f"Анализы: {', '.join(labels[:4])}")
    if not parts:
        return ""
    tail = "Стоит уточнить сроки выполнения."
    return ". ".join(parts) + ". " + tail
