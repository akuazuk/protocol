"""Tests for path-based and enrichment rule loading."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.loader import clear_clinical_knowledge_cache, clinical_knowledge_status
from clinical_knowledge.rules_from_enrichment import enrichment_payload_to_rules, load_enrichment_rules
from clinical_knowledge.rules_from_path import extract_path_rules, infer_path_condition


def test_infer_path_condition_appendicitis():
    sp = "minzdrav_protocols/gastroenterologiya/КП_острым_аппендицитом.pdf"
    got = infer_path_condition(sp)
    assert got is not None
    assert got[0] == "acute_appendicitis"


def test_extract_path_rules_has_source():
    sp = "minzdrav_protocols/gastroenterologiya/КП_холецистит.pdf"
    rules = extract_path_rules(sp, protocol_id="t", rule_id_prefix="abc")
    assert "acute_cholecystitis" in rules
    assert rules["acute_cholecystitis"][0]["extraction_method"] == "path_template"


def test_enrichment_to_rules(tmp_path, monkeypatch):
    enrich_dir = tmp_path / "enrichment"
    enrich_dir.mkdir()
    payload = {
        "condition_id": "acute_cholecystitis",
        "text_hash": "abc123",
        "source_path": "minzdrav_protocols/gastroenterologiya/x.pdf",
        "enrichment": {
            "diagnosis_required_components": ["нозология", "форма", "тяжесть"],
            "diagnostic_criteria_summary": "Критерии из протокола.",
        },
    }
    (enrich_dir / "acute_cholecystitis_abc123.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        "clinical_knowledge.rules_from_enrichment.ENRICH_DIRS",
        (enrich_dir,),
    )
    rules = enrichment_payload_to_rules(payload)
    assert any(r["rule_type"] == "diagnosis_formula" for r in rules)
    loaded = load_enrichment_rules()
    assert "acute_cholecystitis" in loaded


def test_clinical_knowledge_status_has_coverage():
    clear_clinical_knowledge_cache()
    st = clinical_knowledge_status()
    assert "rules_coverage" in st
    assert "pdfs_total" in st["rules_coverage"]
