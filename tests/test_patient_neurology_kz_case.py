"""Golden neurology KZ case (adult female, M53.0)."""
from __future__ import annotations

from pathlib import Path

import pytest

from clinical_knowledge.patient_review import run_patient_review

FIXTURE = Path(__file__).parent / "fixtures" / "neurology_kz_adult.txt"


@pytest.fixture
def neurology_kz() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_neurology_no_pediatric_protocol_in_report(neurology_kz: str) -> None:
    out = run_patient_review(
        text=neurology_kz,
        consultation_id="t-neuro-golden",
        demographics_meta={"age_years": 61, "sex": "female"},
    )
    assert out["ok"] is True
    pr = out["patient_report"]
    blob = str(pr).lower()
    assert "инсульт у дет" not in blob
    assert "pmp22" not in blob
    assert "send_gate" not in blob
    assert "gate_score" not in blob


def test_neurology_v2_summary_and_scores(neurology_kz: str) -> None:
    out = run_patient_review(
        text=neurology_kz,
        consultation_id="t-neuro-v2",
        demographics_meta={"age_years": 61, "sex": "female"},
    )
    pr = out["patient_report"]
    assert pr.get("top_summary")
    assert pr["top_summary"].get("headline_ru")
    assert "срок" in pr["top_summary"]["headline_ru"].lower() or "уточн" in pr["plain_summary_ru"].lower()
    scores = pr.get("scores") or {}
    assert "document_completeness" in scores
    assert "patient_clarity" in scores
    assert "protocol_match_confidence" in scores
    assert pr.get("show_single_overall_score") is False


def test_neurology_meds_and_exams_not_zero(neurology_kz: str) -> None:
    out = run_patient_review(text=neurology_kz, consultation_id="t-neuro-extract")
    pr = out["patient_report"]
    meds = pr.get("extracted_medications") or []
    exams = pr.get("extracted_exams") or []
    assert len(meds) >= 4
    assert len(exams) >= 1
    summary_blob = (pr.get("plain_summary_ru") or "") + (pr.get("medications_summary_ru") or "")
    assert "0 назначений" not in summary_blob.lower()
    assert "0 обследован" not in summary_blob.lower()


def test_neurology_calm_questions_no_forbidden(neurology_kz: str) -> None:
    from clinical_knowledge.patient_questions import is_forbidden_question

    out = run_patient_review(text=neurology_kz, consultation_id="t-neuro-q")
    pr = out["patient_report"]
    for q in pr.get("questions_for_doctor") or []:
        assert not is_forbidden_question(q)
        assert "черновик" not in q.lower()
        assert "половина пропала" not in q.lower()
    assert pr.get("message_to_doctor", {}).get("text_ru")
    assert pr.get("visit_sheet", {}).get("text_ru")


def test_neurology_safe_citations(neurology_kz: str) -> None:
    from clinical_knowledge.patient_quote_quality import is_unsafe_quote

    out = run_patient_review(text=neurology_kz, consultation_id="t-neuro-cites")
    pr = out["patient_report"]
    for c in pr.get("protocol_citations") or []:
        assert not is_unsafe_quote(c.get("excerpt") or "")
