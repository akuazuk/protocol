"""B2C exams block enrich (phase 2b)."""
from __future__ import annotations

from clinical_knowledge.patient_exams_enrich import enrich_lab_crosscheck_with_exams_block


def test_enrich_adds_exams_gap_note() -> None:
    lab_check = {"lab_count": 2, "notes_ru": ["В анализах есть СРБ."]}
    exams_card = {
        "block_id": "exams",
        "score_pct": 40,
        "gaps_ru": ["нет сроков контрольного УЗИ"],
        "comment_ru": "",
    }
    out = enrich_lab_crosscheck_with_exams_block(lab_check, exams_card=exams_card)
    assert out["exams_block_notes_ru"]
    assert "обследований" in out["notes_ru"][0].lower()
    assert "СРБ" in out["notes_ru"][1]


def test_enrich_skips_good_exams_score() -> None:
    lab_check = {"lab_count": 1, "notes_ru": []}
    exams_card = {"block_id": "exams", "score_pct": 90, "gaps_ru": [], "comment_ru": "ОК"}
    out = enrich_lab_crosscheck_with_exams_block(lab_check, exams_card=exams_card)
    assert not out.get("exams_block_notes_ru")
