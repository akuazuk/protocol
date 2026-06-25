"""Извлечение маркеров лабораторных анализов из OCR/PDF текста (B2C)."""
from __future__ import annotations

import re
from typing import Any

# Каноническое имя → паттерны в тексте бланка.
_LAB_MARKERS: list[tuple[str, re.Pattern[str]]] = [
    ("гемоглобин", re.compile(r"гемоглобин|hemoglobin|\bhgb\b", re.I)),
    ("лейкоциты", re.compile(r"лейкоцит|wbc|лейк\.", re.I)),
    ("эритроциты", re.compile(r"эритроцит|rbc", re.I)),
    ("тромбоциты", re.compile(r"тромбоцит|plt|тромб\.", re.I)),
    ("СОЭ", re.compile(r"\bсоэ\b|скорость оседания", re.I)),
    ("глюкоза", re.compile(r"глюкоз|glucose|\bglu\b", re.I)),
    ("СРБ", re.compile(r"\bсрб\b|с-реактивн|crp|c-reactive", re.I)),
    ("АЛТ", re.compile(r"\bалт\b|alt\b|аланинаминотрансфераз", re.I)),
    ("АСТ", re.compile(r"\bast\b|аспартатаминотрансфераз", re.I)),
    ("креатинин", re.compile(r"креатинин|creatinine", re.I)),
    ("мочевина", re.compile(r"мочевин|urea", re.I)),
    ("билирубин", re.compile(r"билирубин|bilirubin", re.I)),
    ("ТТГ", re.compile(r"\bттг\b|тиреотропн|tsh\b", re.I)),
    ("ферритин", re.compile(r"ферритин|ferritin", re.I)),
    ("ОАК", re.compile(r"\bоак\b|общий анализ крови", re.I)),
    ("ОАМ", re.compile(r"\bоам\b|общий анализ мочи", re.I)),
    ("УЗИ", re.compile(r"\bузи\b|ультразвук", re.I)),
    ("рентген", re.compile(r"рентген|рентгенограф|x-ray", re.I)),
    ("КТ", re.compile(r"\bкт\b|компьютерн\w*\s+томограф", re.I)),
    ("МРТ", re.compile(r"\bмрт\b|магнитно-резонанс", re.I)),
    ("ЭКГ", re.compile(r"\bэкг\b|электрокардиограф", re.I)),
]

_VALUE_LINE = re.compile(
    r"(?P<name>[а-яёa-z0-9][а-яёa-z0-9\s\-]{2,40}?)\s*"
    r"(?P<val>\d+[.,]\d+|\d+)\s*"
    r"(?P<unit>%|г/л|ммоль/л|мкмоль/л|×10\^?\d+|10\^?\d+|ед/л|мг/л|мл/мин)?",
    re.I,
)


def extract_lab_markers(text: str) -> list[dict[str, Any]]:
    """Список найденных исследований/показателей в тексте бланка."""
    if not (text or "").strip():
        return []
    blob = re.sub(r"\s+", " ", text)
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for canonical, pat in _LAB_MARKERS:
        if pat.search(blob):
            key = canonical.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append({"marker": canonical, "source": "keyword"})
    for m in _VALUE_LINE.finditer(blob):
        name = (m.group("name") or "").strip()
        if len(name) < 3 or len(name) > 45:
            continue
        key = name.lower()[:40]
        if key in seen:
            continue
        seen.add(key)
        found.append(
            {
                "marker": name[:45],
                "value": m.group("val"),
                "unit": (m.group("unit") or "").strip(),
                "source": "value_line",
            }
        )
        if len(found) >= 40:
            break
    return found


def marker_names(markers: list[dict[str, Any]]) -> list[str]:
    return [str(m.get("marker") or "").strip() for m in markers if m.get("marker")]
