"""Tests for full catalog structuring (gastro-like conditions)."""
from __future__ import annotations

from clinical_knowledge.condition_builder import build_condition_record, merge_condition_records
from clinical_knowledge.catalog_full_build import build_status_payload, load_build_state


def test_build_condition_record_minimal():
    card = {
        "protocol_id": "p1",
        "source_path": "minzdrav_protocols/pulmonologiya-ftiziatriya/pneumonia.pdf",
        "title": "Пневмония у взрослых",
        "icd10_primary": ["J18.9"],
        "population": "adult",
        "specialty_ru": "Пульмонология",
        "approval": {"number": "1", "date": "2020-01-01"},
    }
    rules = [
        {
            "rule_type": "diagnosis_formula",
            "required_components": ["нозология", "локализация", "тяжесть"],
        }
    ]
    rec = build_condition_record("pneumonia", card, rules)
    assert rec["condition_id"] == "pneumonia"
    assert rec["diagnosis_formula"]["required_components"] == ["нозология", "локализация", "тяжесть"]
    assert rec["structured_from"] == "catalog_full_build"


def test_merge_condition_records_icd():
    a = build_condition_record("x", {"title": "A", "source_path": "a.pdf"}, [])
    b = build_condition_record("x", {"title": "B", "source_path": "b.pdf", "icd10_primary": ["K21"]}, [])
    m = merge_condition_records(a, b)
    assert "K21" in (m.get("icd10") or [])


def test_build_status_payload_shape():
    load_build_state()
    payload = build_status_payload()
    assert "build_pct" in payload
    assert "pdfs_total" in payload
