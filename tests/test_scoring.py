"""Тесты scoring (ТЗ §14 / §19)."""
from __future__ import annotations

from clinical_knowledge.consult_schema import ScoreBreakdown
from clinical_knowledge.scoring import compute_overall


def test_eight_block_weighting():
    bd = ScoreBreakdown(
        structural_score=80.0,
        patient_data_score=90.0,
        protocol_match_score=85.0,
        diagnosis_score=75.0,
        required_exams_score=70.0,
        treatment_score=80.0,
        safety_score=100.0,
        follow_up_score=90.0,
        documentation_quality_score=85.0,
    )
    overall, status = compute_overall(bd)
    assert overall is not None
    assert 70 <= overall <= 95
    assert status in ("compliant", "mostly_compliant", "partially_compliant", "non_compliant")


def test_none_blocks_renormalized():
    bd = ScoreBreakdown(
        structural_score=100.0,
        patient_data_score=100.0,
        diagnosis_score=100.0,
        treatment_score=100.0,
    )
    overall, status = compute_overall(bd)
    assert overall == 100.0
    assert status == "compliant"


def test_manual_review_override():
    bd = ScoreBreakdown(
        structural_score=95.0,
        diagnosis_score=95.0,
        treatment_score=95.0,
        safety_score=0.0,
    )
    overall, status = compute_overall(bd, force_manual_review=True)
    assert overall is not None
    assert status == "manual_review_required"
