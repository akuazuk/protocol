"""Дополнение отчёта замечаниями по блоку «обследования» в КЗ (B2C, фаза 2b)."""
from __future__ import annotations

from typing import Any


def exams_block_notes_for_report(
    *,
    exams_card: dict[str, Any] | None,
) -> list[str]:
    """Замечания по разделу обследований в самом заключении (не по бланкам анализов)."""
    notes: list[str] = []
    if not exams_card or not isinstance(exams_card, dict):
        return notes

    score = exams_card.get("score_pct")
    gaps = [str(g).strip() for g in (exams_card.get("gaps_ru") or []) if str(g).strip()]
    summary = str(exams_card.get("comment_ru") or "").strip()

    if isinstance(score, (int, float)) and float(score) < 75:
        if gaps:
            notes.append(
                "В разделе «Обследования» заключения не хватает деталей: "
                + "; ".join(gaps[:3])
                + ". Спросите врача, какие исследования уже сделаны и что планируется."
            )
        elif summary and len(summary) > 16:
            notes.append(f"По обследованиям в заключении: {summary}")
        else:
            notes.append(
                "Раздел обследований в заключении описан кратко. "
                "Уточните у врача план диагностики и сроки."
            )
    elif gaps:
        notes.append("По обследованиям в заключении: " + gaps[0])
    return notes
