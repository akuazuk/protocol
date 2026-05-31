"""Тесты архива снимков анализа КЗ."""
from __future__ import annotations

import json

from clinical_knowledge.analysis_archive import build_snapshot, load_snapshots, save_snapshot


def test_build_snapshot_anonymized(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSULT_ARCHIVE_DIR", str(tmp_path))
    monkeypatch.setenv("CONSULT_ARCHIVE_ANALYSES", "1")

    sa = {
        "document": {
            "patient": {"full_name": "Кузавка Павел Леонидович", "age_years": 48, "sex": "male"},
            "doctor_specialty": "Флеболог",
        },
        "compliance": {
            "overall_status": "mostly_compliant",
            "overall_score": 82.0,
            "score_breakdown": {"diagnosis_score": 77.0},
            "diagnosis_assessments": [{"icd10_code": "I80.1", "status": "supported", "diagnosis_text": "x"}],
            "safety_assessments": [{}],
        },
        "rubric_specifics": {"rubrics": ["bolezni-sistemy-krovoobrashcheniya"]},
    }
    snap = build_snapshot(
        full_text="тестовый текст консультации",
        source_file="pl_1_f.pdf",
        build_version="test",
        structured_analysis=sa,
        icd_codes=["I80.1"],
    )
    assert snap["patient_initials"] == "К. П. Л."
    assert "full_name" not in json.dumps(snap)
    assert snap["source_basename"] == "pl_1_f.pdf"

    path = save_snapshot(snap)
    assert path is not None
    loaded = load_snapshots()
    assert len(loaded) == 1
    assert loaded[0]["text_hash"] == snap["text_hash"]


def test_archive_off_by_default(monkeypatch):
    monkeypatch.delenv("CONSULT_ARCHIVE_ANALYSES", raising=False)
    from clinical_knowledge.analysis_archive import archive_enabled

    assert archive_enabled() is False
