"""Таблица случаев: clinical_visit + legacy consultation."""
from __future__ import annotations

from clinical_knowledge.mo_backend import (
    _apply_score_eligible_default,
    is_case_score_eligible,
)


def test_cases_hard_clinical_kinds_only() -> None:
    out = _apply_score_eligible_default({})
    assert out["document_kinds"] == "clinical_visit|consultation"
    assert out["score_eligible_only"] == "1"


def test_cases_ignore_opt_out_all_kinds() -> None:
    out = _apply_score_eligible_default({"score_eligible_only": "0"})
    assert "clinical_visit" in out["document_kinds"]
    assert out["score_eligible_only"] == "1"


def test_cases_ignore_nonclinical_document_kinds() -> None:
    out = _apply_score_eligible_default(
        {"document_kinds": "procedure_session|medical_exam", "score_eligible_only": "1"}
    )
    assert "procedure_session" not in out["document_kinds"]
    assert "clinical_visit" in out["document_kinds"]


def test_is_case_score_eligible_accepts_legacy_consultation() -> None:
    assert is_case_score_eligible({"document_kind": "clinical_visit"})
    assert is_case_score_eligible({"document_kind": "consultation"})
    assert is_case_score_eligible(
        {"document_kind": "clinical_visit"},
        document_kinds=["consultation", "clinical_visit"],
    )
    # document legacy + warehouse clinical → eligible
    assert is_case_score_eligible(
        {"document_kind": "clinical_visit"},
        document_kinds=["consultation"],
    )
    assert not is_case_score_eligible({"document_kind": "procedure_session"})
    assert not is_case_score_eligible({"document_kind": "medical_exam"})
    assert not is_case_score_eligible(document_kind="diagnostic")
