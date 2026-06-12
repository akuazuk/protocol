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
