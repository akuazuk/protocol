"""Классификация chunk_type по разделу и тексту протокола."""
from __future__ import annotations

import re
from typing import Any

# Приоритетные заголовки разделов (точное совпадение начала)
_SECTION_HEADING_TYPES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*(?:\d+\.)?\s*диагностик", re.I), "diagnostics"),
    (re.compile(r"^\s*(?:\d+\.)?\s*лечени", re.I), "treatment"),
    (re.compile(r"^\s*(?:\d+\.)?\s*профилактик", re.I), "prevention"),
    (re.compile(r"^\s*(?:\d+\.)?\s*реабилитац", re.I), "rehabilitation"),
    (re.compile(r"^\s*(?:\d+\.)?\s*диспансерн", re.I), "dispensary"),
    (re.compile(r"^\s*(?:\d+\.)?\s*классификац", re.I), "classification"),
    (re.compile(r"^\s*(?:\d+\.)?\s*маршрутизац", re.I), "routing"),
    (re.compile(r"^\s*(?:\d+\.)?\s*фармакотерап", re.I), "pharmacotherapy"),
    (re.compile(r"^\s*(?:\d+\.)?\s*алгоритм", re.I), "algorithm"),
    (re.compile(r"^\s*(?:\d+\.)?\s*приложени", re.I), "appendix"),
    (re.compile(r"^\s*(?:\d+\.)?\s*термин", re.I), "terms"),
    (re.compile(r"^\s*(?:\d+\.)?\s*общие\s+положени", re.I), "body"),
    (re.compile(r"^\s*(?:\d+\.)?\s*показани", re.I), "criteria_block"),
    (re.compile(r"^\s*(?:\d+\.)?\s*противопоказани", re.I), "criteria_block"),
    (re.compile(r"^\s*(?:\d+\.)?\s*критери", re.I), "criteria_block"),
]

# Regex по combined text (порядок важен: более специфичные раньше)
_TEXT_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"фармакотерап|медикаментозн", re.I), "pharmacotherapy"),
    (re.compile(r"лекарственн(?:ые|\s+средств)|препарат|доз(?:а|ировк)", re.I), "drug_list"),
    (re.compile(r"показани(?:я|й)\s+(?:к\s+)?|противопоказани", re.I), "criteria_block"),
    (re.compile(r"диагностик|обследован|лабораторн|инструментальн", re.I), "diagnostics"),
    (re.compile(r"лечени|терапи|хирург|операц", re.I), "treatment"),
    (re.compile(r"профилактик", re.I), "prevention"),
    (re.compile(r"реабилитац", re.I), "rehabilitation"),
    (re.compile(r"диспансерн|наблюден", re.I), "dispensary"),
    (re.compile(r"классификац|шифр(?:ы)?\s+мкб", re.I), "classification"),
    (re.compile(r"маршрутизац|госпитализац|направлени", re.I), "routing"),
    (re.compile(r"алгоритм|схем(?:а|ы)", re.I), "algorithm"),
    (re.compile(r"термин|определени|понятие", re.I), "terms"),
]

_WEAK_SECTION_TITLES = frozenset({
    "таблица",
    "постановляет:",
    "утверждено",
    "согласовано",
    "документ",
    "№ 2435-xii",
})


def _type_from_section_heading(title: str) -> str | None:
    t = (title or "").strip()
    if not t:
        return None
    for pat, ctype in _SECTION_HEADING_TYPES:
        if pat.search(t):
            return ctype
    return None


def _type_from_section_path(section_path: list[str] | None) -> str | None:
    if not section_path:
        return None
    for label in reversed(section_path):
        hit = _type_from_section_heading(str(label))
        if hit and hit != "body":
            return hit
    return None


def _type_from_text(text: str) -> str | None:
    blob = (text or "").strip()
    if not blob:
        return None
    for pat, ctype in _TEXT_TYPE_PATTERNS:
        if pat.search(blob):
            return ctype
    return None


def infer_chunk_type(
    *,
    section_title: str = "",
    section_number: str = "",
    section_path: list[str] | None = None,
    text: str = "",
    fallback: str = "body",
) -> str:
    """Определить chunk_type: section heading > section_path > full text > fallback."""
    for candidate in (
        section_title,
        section_number,
    ):
        hit = _type_from_section_heading(str(candidate))
        if hit and hit != "body":
            return hit

    hit = _type_from_section_path(section_path)
    if hit:
        return hit

    hit = _type_from_text(text)
    if hit:
        return hit

    return fallback


def resolve_section_title(
    section_title: str,
    section_path: list[str] | None = None,
) -> str:
    """Заменить слабый section_title родительским из section_path."""
    st = (section_title or "").strip()
    low = st.lower()
    if low and low not in _WEAK_SECTION_TITLES and len(st) >= 8:
        return st
    if section_path:
        for label in reversed(section_path):
            lbl = str(label).strip()
            if lbl and lbl.lower() not in _WEAK_SECTION_TITLES and len(lbl) >= 8:
                return lbl
    return st or "Текст протокола"


def guess_chunk_type(section_title: str, text: str) -> str:
    """Обратная совместимость с build_rich_chunks."""
    return infer_chunk_type(section_title=section_title, text=text)
