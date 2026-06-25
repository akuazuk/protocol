"""Дополнение сверки анализов блоком «обследования» из alignment (B2C, фаза 2b)."""
from __future__ import annotations

from typing import Any


def enrich_lab_crosscheck_with_exams_block(
    lab_check: dict[str, Any],
    *,
    exams_card: dict[str, Any] | None,
) -> dict[str, Any]:
    """Добавить замечания по structured-блоку обследований."""
    out = dict(lab_check or {})
    notes = list(out.get("notes_ru") or [])
    exams_notes: list[str] = []

    if exams_card and isinstance(exams_card, dict):
        score = exams_card.get("score_pct")
        gaps = [str(g).strip() for g in (exams_card.get("gaps_ru") or []) if str(g).strip()]
        summary = str(exams_card.get("comment_ru") or "").strip()
        if isinstance(score, (int, float)) and float(score) < 75:
            if gaps:
                exams_notes.append(
                    "В разделе обследований заключения не хватает деталей: "
                    + "; ".join(gaps[:3])
                    + ". Спросите врача, какие исследования уже сделаны и что планируется."
                )
            elif summary and len(summary) > 16:
                exams_notes.append(f"По обследованиям: {summary}")
            else:
                exams_notes.append(
                    "Раздел обследований в заключении описан кратко. "
                    "Уточните у врача план диагностики и сроки."
                )
        elif gaps:
            exams_notes.append("По обследованиям: " + gaps[0])

    for n in exams_notes:
        if n and n not in notes:
            notes.insert(0, n)

    out["notes_ru"] = notes
    out["exams_block_notes_ru"] = exams_notes
    return out
