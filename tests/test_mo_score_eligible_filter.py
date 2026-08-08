"""Таблица случаев жёстко только clinical_visit."""
from __future__ import annotations

from clinical_knowledge.mo_backend import (
    _apply_score_eligible_default,
    is_case_score_eligible,
)


def test_cases_hard_clinical_visit_only() -> None:
    out = _apply_score_eligible_default({})
    assert out["document_kinds"] == "clinical_visit"
    assert out["score_eligible_only"] == "1"


def test_cases_ignore_opt_out_all_kinds() -> None:
    out = _apply_score_eligible_default({"score_eligible_only": "0"})
    assert out["document_kinds"] == "clinical_visit"
    assert out["score_eligible_only"] == "1"


def test_cases_ignore_nonclinical_document_kinds() -> None:
    out = _apply_score_eligible_default(
        {"document_kinds": "procedure_session|medical_exam", "score_eligible_only": "1"}
    )
    assert out["document_kinds"] == "clinical_visit"


def test_is_case_score_eligible_only_clinical_visit() -> None:
    assert is_case_score_eligible({"document_kind": "clinical_visit"})
    assert not is_case_score_eligible({"document_kind": "procedure_session"})
    assert not is_case_score_eligible({"document_kind": "medical_exam"})
    assert not is_case_score_eligible({"document_kind": "non_clinical"})
    assert not is_case_score_eligible(document_kind="diagnostic")
