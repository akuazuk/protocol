"""Сверка загруженных анализов с текстом КЗ (B2C, фаза 2)."""
from __future__ import annotations

import re
from typing import Any

from .lab_result_parser import detect_lab_panels, extract_lab_markers, format_marker_line


def _normalize_marker(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _marker_in_text(marker: str, text: str) -> bool:
    m = _normalize_marker(marker)
    if not m or not text:
        return False
    kz = text.lower()
    if m in kz:
        return True
    # Синонимы для типовых показателей.
    aliases: dict[str, tuple[str, ...]] = {
        "мочевина": ("urea",),
        "креатинин": ("crea", "creatinine"),
        "глюкоза": ("gluc", "glucose"),
        "билирубин общий": ("билирубин", "t-bil", "bil"),
        "холестерин общий": ("холестерин", "chol"),
        "общий белок": ("белок", "tprot"),
        "срб": ("crp", "c-реактивн"),
        "аст": ("ast", "аспартат"),
        "алт": ("alt",),
    }
    for alt in aliases.get(m, ()):
        if alt in kz:
            return True
    if len(m) <= 4:
        return bool(re.search(rf"\b{re.escape(m)}\b", text, re.I))
    return False


def _structured_markers(markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Только строки с числовым/качественным результатом, без мусора OCR."""
    out = [m for m in markers if m.get("value") not in (None, "")]
    if out:
        return out
    return markers


def crosscheck_labs_with_kz(
    *,
    kz_text: str,
    lab_text: str,
) -> dict[str, Any]:
    """Сравнить показатели из бланков анализов с упоминаниями в КЗ."""
    kz = (kz_text or "").strip()
    lab = (lab_text or "").strip()
    empty = {
        "lab_count": 0,
        "panels_ru": [],
        "summary_ru": "",
        "markers": [],
        "markers_table": [],
        "marker_lines_ru": [],
        "in_kz": [],
        "missing_in_kz": [],
        "in_kz_lines": [],
        "missing_in_kz_lines": [],
        "notes_ru": [],
    }
    if not lab:
        return empty

    raw_markers = extract_lab_markers(lab)
    markers = _structured_markers(raw_markers)
    panels = detect_lab_panels(lab)

    table: list[dict[str, Any]] = []
    in_kz: list[str] = []
    missing: list[str] = []
    in_lines: list[str] = []
    miss_lines: list[str] = []

    for m in markers:
        name = str(m.get("marker") or "").strip()
        if not name:
            continue
        line = format_marker_line(m)
        in_kz_flag = _marker_in_text(name, kz)
        row = {
            "marker": name,
            "value": m.get("value"),
            "unit": m.get("unit") or "",
            "flag": m.get("flag") or "",
            "line_ru": line,
            "in_kz": in_kz_flag,
        }
        table.append(row)
        if in_kz_flag:
            in_kz.append(name)
            in_lines.append(line)
        else:
            missing.append(name)
            miss_lines.append(line)

    notes: list[str] = []
    summary = ""
    if not table:
        summary = (
            "Не удалось надёжно распознать показатели в загруженных бланках. "
            "Попробуйте PDF из лаборатории или более чёткое фото."
        )
        notes.append(summary)
    elif panels:
        summary = f"Из бланков ({', '.join(panels[:2])}) прочитано {len(table)} показателей."
        if miss_lines:
            summary += f" {len(miss_lines)} не названы в тексте заключения."
        elif in_lines:
            summary += " Основные показатели упомянуты в заключении."
    else:
        summary = f"Прочитано {len(table)} показателей из загруженных бланков."
        if miss_lines:
            summary += f" {len(miss_lines)} не отражены в тексте заключения."

    if miss_lines:
        notes.append(
            "Показатели из бланков, которых нет в тексте заключения: "
            + "; ".join(miss_lines[:5])
            + (f" и ещё {len(miss_lines) - 5}" if len(miss_lines) > 5 else "")
            + ". Уточните у врача, учтены ли они."
        )

    return {
        "lab_count": len(table),
        "panels_ru": panels[:5],
        "summary_ru": summary,
        "markers": markers[:25],
        "markers_table": table[:25],
        "marker_lines_ru": [r["line_ru"] for r in table[:20]],
        "in_kz": in_kz[:15],
        "missing_in_kz": missing[:15],
        "in_kz_lines": in_lines[:15],
        "missing_in_kz_lines": miss_lines[:15],
        "notes_ru": notes,
    }
