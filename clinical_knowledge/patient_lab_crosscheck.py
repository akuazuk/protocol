"""Сверка загруженных анализов с текстом КЗ (B2C, фаза 2)."""
from __future__ import annotations

import re
from typing import Any

from .lab_result_parser import detect_lab_panels, extract_lab_markers, format_marker_line, marker_names


def _normalize_marker(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _marker_in_text(marker: str, text: str) -> bool:
    m = _normalize_marker(marker)
    if not m or not text:
        return False
    if m in text.lower():
        return True
    # Короткие аббревиатуры - граница слова.
    if len(m) <= 4:
        return bool(re.search(rf"\b{re.escape(m)}\b", text, re.I))
    return False


def crosscheck_labs_with_kz(
    *,
    kz_text: str,
    lab_text: str,
) -> dict[str, Any]:
    """Сравнить маркеры из бланков анализов с упоминаниями в КЗ."""
    kz = (kz_text or "").strip()
    lab = (lab_text or "").strip()
    if not lab:
        return {
            "lab_count": 0,
            "markers": [],
            "in_kz": [],
            "missing_in_kz": [],
            "notes_ru": [],
        }

    markers = extract_lab_markers(lab)
    names = marker_names(markers)
    panels = detect_lab_panels(lab)
    marker_lines = [format_marker_line(m) for m in markers[:20]]
    in_kz: list[str] = []
    missing: list[str] = []
    for name in names:
        if _marker_in_text(name, kz):
            in_kz.append(name)
        else:
            missing.append(name)

    notes: list[str] = []
    if panels:
        notes.append("Распознаны бланки: " + ", ".join(panels[:3]) + ".")
    if marker_lines:
        preview = "; ".join(marker_lines[:6])
        if len(marker_lines) > 6:
            preview += f" и ещё {len(marker_lines) - 6}"
        notes.append(f"Найденные показатели: {preview}.")
    if missing:
        shown = ", ".join(missing[:5])
        if len(missing) > 5:
            shown += f" и ещё {len(missing) - 5}"
        notes.append(
            f"В загруженных анализах найдены показатели ({shown}), "
            "которые не отражены в тексте заключения. Уточните у врача, учтены ли они."
        )
    if names and not missing:
        notes.append("Основные показатели из анализов упомянуты или согласуются с заключением.")
    elif not names:
        notes.append(
            "Не удалось надёжно распознать показатели в бланке анализов. "
            "Проверьте качество фото или загрузите текстовый PDF."
        )

    return {
        "lab_count": len(markers),
        "panels_ru": panels[:5],
        "markers": markers[:25],
        "marker_lines_ru": marker_lines[:20],
        "in_kz": in_kz[:15],
        "missing_in_kz": missing[:15],
        "notes_ru": notes,
    }
