"""Извлечение сущностей из текста: МКБ-10, группы населения, ЛС, сроки."""
from __future__ import annotations

import re
from typing import Any

# МКБ-10: латинская буква + 2 цифры, опционально .подрубрика
ICD10_RE = re.compile(
    r"\b([A-Z]\d{2}(?:\.\d{1,4})?)\b",
    re.I,
)

POPULATION_MARKERS = [
    ("новорожд", "новорождённые"),
    ("детск", "дети"),
    ("детей", "дети"),
    ("ребён", "дети"),
    ("ребен", "дети"),
    ("подрост", "подростки"),
    ("взросл", "взрослые"),
    ("беремен", "беременные"),
    ("женщин", "женщины"),
    ("мужчин", "мужчины"),
    ("пожил", "пожилые"),
]

CARE_MARKERS = [
    ("амбулатор", "амбулаторно"),
    ("стационар", "стационар"),
    ("I уров", "I уровень"),
    ("II уров", "II уровень"),
    ("III уров", "III уровень"),
    ("IV уров", "IV уровень"),
    ("скорой", "скорая помощь"),
    ("неотложн", "неотложная помощь"),
]

# Простой захват названий препаратов (группы / МНН) — без гарантии полноты
DRUG_LINE_RE = re.compile(
    r"(?:препарат|назначают|применяют|терапия|лечение)[^.]{0,120}?([А-ЯЁA-Z][а-яёa-z\-]+(?:\s+[а-яёa-z\-]+){0,4})",
    re.I,
)

DURATION_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(суток|сут|дн(?:ей|я)?|недел(?:ь|и|е)|мес(?:яц(?:ев|а)?)?|лет|год|час(?:ов|а)?|мин(?:ут)?)",
    re.I,
)


def extract_icd10(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in ICD10_RE.finditer(text or ""):
        code = m.group(1).upper().replace(" ", "")
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


def extract_populations(text: str) -> list[str]:
    low = (text or "").lower()
    seen: set[str] = set()
    out: list[str] = []
    for needle, label in POPULATION_MARKERS:
        if needle in low and label not in seen:
            seen.add(label)
            out.append(label)
    return out


def extract_care_settings(text: str) -> list[str]:
    low = (text or "").lower()
    seen: set[str] = set()
    out: list[str] = []
    for needle, label in CARE_MARKERS:
        if needle in low and label not in seen:
            seen.add(label)
            out.append(label)
    return out


def extract_conditions_snippets(text: str, max_n: int = 12) -> list[str]:
    """Короткие фразы с критериями/показаниями (эвристика)."""
    lines = []
    for line in (text or "").split("\n"):
        low = line.lower()
        if any(
            x in low
            for x in ("показан", "противопоказ", "критер", "диагноз", "состояни")
        ):
            t = line.strip()
            if 20 < len(t) < 400:
                lines.append(t)
        if len(lines) >= max_n:
            break
    return lines


def extract_drugs_heuristic(text: str, max_n: int = 40) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in DRUG_LINE_RE.finditer(text or ""):
        g = m.group(1).strip()
        if len(g) > 3 and g not in seen:
            seen.add(g)
            out.append(g)
        if len(out) >= max_n:
            break
    return out


def extract_durations(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in DURATION_RE.finditer(text or ""):
        s = m.group(0).strip()
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def keywords_from_text(text: str, max_kw: int = 30) -> list[str]:
    """Простые ключевые слова (длинные токены)."""
    words = re.findall(r"[а-яА-ЯёЁ]{5,}", text or "")
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        low = w.lower()
        if low not in seen:
            seen.add(low)
            out.append(w.lower())
        if len(out) >= max_kw:
            break
    return out


def build_embedding_ready_text(
    section_title: str, chunk_text: str, icd10: list[str], pops: list[str]
) -> str:
    parts = []
    if section_title:
        parts.append(section_title)
    if icd10:
        parts.append("МКБ-10: " + ", ".join(icd10[:12]))
    if pops:
        parts.append("Популяция: " + ", ".join(pops[:8]))
    parts.append(chunk_text)
    return "\n".join(parts)
