"""Tests for strict consult protocol path selection."""
from __future__ import annotations

from clinical_knowledge.consult_retrieval import (
    consult_target_protocol_paths,
    filter_retrieval_by_category_slugs,
    filter_retrieval_rows_by_paths,
)


def test_filter_retrieval_rows_by_paths():
    rows = [
        {"path": "minzdrav_protocols/pulmonologiya/a.pdf", "score": 1.0},
        {"path": "minzdrav_protocols/stomatologiya/b.pdf", "score": 0.9},
    ]
    out = filter_retrieval_rows_by_paths(
        rows, ["minzdrav_protocols/pulmonologiya/a.pdf"]
    )
    assert len(out) == 1
    assert "pulmonologiya" in out[0]["path"]


def test_consult_target_from_matched_rules():
    rules = {
        "matched_protocols": [
            {"source_path": "minzdrav_protocols/gastroenterologiya/gerd.pdf", "match_score": 80}
        ]
    }
    paths, meta = consult_target_protocol_paths(
        merged_icd=["K21.9"],
        diag_icd=["K21.9"],
        clinical_rules=rules,
        specialty_slugs=["gastroenterologiya"],
    )
    assert paths
    assert "gerd.pdf" in paths[0]
    assert meta.get("strict")


def test_filter_retrieval_by_category_slugs():
    rows = [
        {"path": "minzdrav_protocols/nevrologiya-neyrokhirurgiya/a.pdf", "category": "nevrologiya-neyrokhirurgiya"},
        {"path": "minzdrav_protocols/akusherstvo-ginekologiya/b.pdf", "category": "akusherstvo-ginekologiya"},
    ]
    out = filter_retrieval_by_category_slugs(
        rows,
        ["nevrologiya-neyrokhirurgiya"],
        strict=True,
    )
    assert len(out) == 1
    assert out[0]["category"] == "nevrologiya-neyrokhirurgiya"


def test_m54_always_has_protocol_pick():
    facts = {
        "consultation": {
            "complaints": ["боль в пояснице с иррадиацией в ногу"],
            "diagnosis_text": "M54.3 ишиас",
            "conditions_hint": ["боль в пояснице"],
            "performed_exams": [],
        },
        "patient_context": {"adult_or_child": "adult"},
    }
    paths, meta = consult_target_protocol_paths(
        merged_icd=["M54.3"],
        diag_icd=["M54.3"],
        clinical_rules=None,
        specialty_slugs=["nevrologiya"],
        consult_facts=facts,
        primary_specialty="nevrologiya",
        min_match_score=22.0,
    )
    assert paths, meta
    top = (meta.get("protocol_matches") or [{}])[0]
    assert float(top.get("match_score") or 0) >= 12.0
    assert "nevrologiya-neyrokhirurgiya" in str(paths[0]) or meta.get("icd_coverage_fallback")
