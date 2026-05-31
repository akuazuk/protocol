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
    (re.compile(r"формулировк[аи]\s+диагноз", re.I), "diagnosis_formula", "Формулировка диагноза"),
    (re.compile(r"диагностическ(?:им|ими)\s+критери", re.I), "diagnostic_criteria", "Диагностические критерии"),
    (re.compile(r"клиническ(?:ие|их)\s+критери", re.I), "clinical_criteria", "Клинические критерии"),
    (re.compile(r"показан(?:ия|ий)\s+к\s+обслед", re.I), "exam_indications", "Показания к обследованию"),
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


# Нумерация заголовка в начале строки: "ГЛАВА 3", "3.", "3.1.", "3.1.2)" и т.п.
SECTION_NUMBER_RE = re.compile(
    r"^\s*(?:глава\s+(\d+)|((?:\d+\.)+\d*|\d+)[.)])(?=\s|$)",
    re.I,
)


def _make_id(doc_id: str, idx: int) -> str:
    return f"{doc_id}_sec_{idx}"


def _extract_section_number(head_line: str) -> str:
    """Возвращает нормализованный номер раздела (например '3' или '3.1') или ''."""
    m = SECTION_NUMBER_RE.match(head_line)
    if not m:
        return ""
    if m.group(1):  # "ГЛАВА N"
        return m.group(1)
    raw = (m.group(2) or "").strip().strip(".")
    return raw


def _section_title(head_line: str, label: str) -> str:
    """Читаемый заголовок раздела: первая строка заголовка, иначе типовая метка."""
    title = (head_line or "").strip()
    # Отрезаем хвост, если строка слишком длинная (заголовок + начало текста).
    if len(title) > 120:
        title = title[:120].rstrip()
    return title or label


def _build_section_path(section_number: str, label: str, number_to_label: dict[str, str]) -> list[str]:
    """Иерархический путь по числовой вложенности: '3.1' -> [label(3), label(3.1)]."""
    if not section_number:
        return [label]
    parts = section_number.split(".")
    path: list[str] = []
    for depth in range(1, len(parts) + 1):
        prefix = ".".join(parts[:depth])
        path.append(number_to_label.get(prefix, label if depth == len(parts) else prefix))
    return path or [label]


def detect_sections(doc_id: str, text: str) -> list[dict[str, Any]]:
    """
    Разбиение по заголовкам разделов; для нумерованных заголовков строится
    иерархический section_path и сохраняется section_number/section_title.
    """
    if not text:
        return []
    lines = text.split("\n")
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

    # Первый проход: карта номер -> читаемая метка для построения иерархии.
    number_to_label: dict[str, str] = {}
    for _start, _li, _stype, label, head_line in hits:
        num = _extract_section_number(head_line)
        if num and num not in number_to_label:
            number_to_label[num] = _section_title(head_line, label)

    sections: list[dict[str, Any]] = []
    for i, (start, _li, stype, label, head_line) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        chunk = text[start:end].strip()
        if len(chunk) < 30:
            continue
        sec_id = _make_id(doc_id, i)
        section_number = _extract_section_number(head_line)
        section_title = _section_title(head_line, label)
        section_path = _build_section_path(section_number, section_title, number_to_label)
        sections.append(
            {
                "section_id": sec_id,
                "section_type": stype,
                "label": label,
                "section_number": section_number,
                "section_title": section_title,
                "head_line": head_line,
                "start_char": start,
                "end_char": end,
                "text": chunk,
                "section_path": section_path,
            }
        )

    if not sections:
        sections.append(
            {
                "section_id": _make_id(doc_id, 0),
                "section_type": "body",
                "label": "Документ",
                "section_number": "",
                "section_title": "Документ",
                "head_line": "",
                "start_char": 0,
                "end_char": len(text),
                "text": text,
                "section_path": ["Документ"],
            }
        )
    return sections
