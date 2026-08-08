"""Таблица случаев по умолчанию только clinical_visit."""
from __future__ import annotations

from clinical_knowledge.mo_backend import _apply_score_eligible_default


def test_cases_default_to_clinical_visit_only() -> None:
    out = _apply_score_eligible_default({})
    assert out["document_kinds"] == "clinical_visit"
    assert out["score_eligible_only"] == "1"


def test_cases_can_show_all_document_kinds() -> None:
    out = _apply_score_eligible_default({"score_eligible_only": "0"})
    assert "document_kinds" not in out or not out.get("document_kinds")


def test_explicit_document_kinds_wins() -> None:
    out = _apply_score_eligible_default(
        {"document_kinds": "procedure_session|medical_exam", "score_eligible_only": "1"}
    )
    assert out["document_kinds"] == "procedure_session|medical_exam"
