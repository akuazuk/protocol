"""Обнаружение типовых разделов клинических протоколов (регексы по-русски)."""
from __future__ import annotations

import re
from typing import Any

# (regex, section_type, human_label)
SECTION_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"^\s*(?:преамбула|вступлени)", re.I), "preamble", "Преамбула"),
    (
        re.compile(r"общие\s+положени", re.I),
        "general_provisions",
        "Общие положения",
    ),
    (
        re.compile(r"термины\s+и\s+определени", re.I),
        "terms",
        "Термины и определения",
    ),
    (re.compile(r"классификаци|стадии", re.I), "classification", "Классификация"),
    (re.compile(r"диагностик", re.I), "diagnostics", "Диагностика"),
    (re.compile(r"лечени|терапи", re.I), "treatment", "Лечение"),
    (re.compile(r"профилактик", re.I), "prevention", "Профилактика"),
    (
        re.compile(r"реабилитаци", re.I),
        "rehabilitation",
        "Медицинская реабилитация",
    ),
    (
        re.compile(r"диспансерн|наблюден", re.I),
        "dispensary",
        "Диспансерное наблюдение",
    ),
    (re.compile(r"маршрутизаци|госпитализац", re.I), "routing", "Маршрутизация"),
    (re.compile(r"фармакотерапи|медикамент", re.I), "pharmacotherapy", "Фармакотерапия"),
    (re.compile(r"приложени", re.I), "appendix", "Приложения"),
    (re.compile(r"таблиц", re.I), "tables", "Таблицы"),
    (re.compile(r"алгоритм", re.I), "algorithm", "Алгоритмы"),
]


def _make_id(doc_id: str, idx: int) -> str:
    return f"{doc_id}_sec_{idx}"


def detect_sections(doc_id: str, text: str) -> list[dict[str, Any]]:
    """
    Грубое разбиение: ищем заголовки разделов по строкам.
    Возвращает плоский список участков с section_path.
    """
    if not text:
        return []
    lines = text.split("\n")
    # Собираем позиции строк в тексте
    line_starts: list[int] = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1

    hits: list[tuple[int, int, str, str, str]] = []
    for li, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < 8:
            continue
        for rx, stype, label in SECTION_RULES:
            if rx.search(stripped[:120]):
                start = line_starts[li]
                hits.append((start, li, stype, label, stripped[:200]))
                break

    hits.sort(key=lambda x: x[0])
    sections: list[dict[str, Any]] = []
    for i, (start, _li, stype, label, head_line) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        chunk = text[start:end].strip()
        if len(chunk) < 30:
            continue
        sec_id = _make_id(doc_id, i)
        sections.append(
            {
                "section_id": sec_id,
                "section_type": stype,
                "label": label,
                "head_line": head_line,
                "start_char": start,
                "end_char": end,
                "text": chunk,
                "section_path": [label],
            }
        )

    if not sections:
        sections.append(
            {
                "section_id": _make_id(doc_id, 0),
                "section_type": "body",
                "label": "Документ",
                "head_line": "",
                "start_char": 0,
                "end_char": len(text),
                "text": text,
                "section_path": ["Документ"],
            }
        )
    return sections
