"""B2C exams block notes (phase 2b)."""
from __future__ import annotations

from clinical_knowledge.patient_exams_enrich import exams_block_notes_for_report


def test_exams_block_notes_from_gaps() -> None:
    exams_card = {
        "block_id": "exams",
        "score_pct": 40,
        "gaps_ru": ["нет сроков контрольного УЗИ"],
        "comment_ru": "",
    }
    notes = exams_block_notes_for_report(exams_card=exams_card)
    assert notes
    assert "обследован" in notes[0].lower()
    assert "узи" in notes[0].lower()


def test_exams_block_notes_skip_good_score() -> None:
    exams_card = {"block_id": "exams", "score_pct": 90, "gaps_ru": [], "comment_ru": "ОК"}
    assert not exams_block_notes_for_report(exams_card=exams_card)
