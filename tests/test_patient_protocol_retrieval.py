"""B2C RAG retrieval for patient review."""
from __future__ import annotations

import rag_server as rs
from clinical_knowledge.patient_protocol_retrieval import (
    patient_protocol_citations_from_retrieved,
    retrieve_patient_protocol_context,
)


def test_formats_hint_single_file() -> None:
    hint = rs.consult_review_formats_hint_ru(max_files=1)
    assert "Один файл" in hint
    assert "до 5" not in hint


def test_citations_from_retrieved_dedupes() -> None:
    rows = [
        {"text": "Пациенту рекомендовано обследование " * 3, "section_title": "Диагностика", "path": "a.pdf"},
        {"text": "Пациенту рекомендовано обследование " * 3, "section_title": "Дубль", "path": "a.pdf"},
    ]
    cites = patient_protocol_citations_from_retrieved(rows, limit=3)
    assert len(cites) == 1
    assert cites[0]["title"] == "Диагностика"


def test_retrieve_patient_protocol_context_empty_without_text(monkeypatch) -> None:
    monkeypatch.setenv("PATIENT_RAG_RETRIEVAL_ENABLED", "0")
    out = retrieve_patient_protocol_context(kz_text="")
    assert out["paths"] == []
    assert out["rag_used"] is False
