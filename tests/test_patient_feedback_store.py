"""Patient feedback store and quality flags."""
from __future__ import annotations

from clinical_knowledge.patient_feedback_store import compute_quality_flags, text_hash


def test_text_hash_stable() -> None:
    h1 = text_hash("hello")
    h2 = text_hash("hello")
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_flag_mri_hallucination() -> None:
    report = {
        "plain_summary_ru": "Стоит уточнить сроки МРТ",
        "extracted_exams": [{"exam_type": "LAB_OAK", "category": "lab"}],
        "understood_from_document": [],
        "patient_context": {},
    }
    flags = compute_quality_flags(kz_text="ОАК назначен", report=report)
    assert "no_mri_in_source_but_in_summary" in flags
