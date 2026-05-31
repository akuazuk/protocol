"""Tests for rule filtering by matched protocol."""
from __future__ import annotations

from clinical_knowledge.loader import clear_clinical_knowledge_cache
from clinical_knowledge.rule_filter import filter_rules_for_matched_protocols


def test_filter_keeps_manual_always():
    rules = [
        {"rule_id": "manual", "rule_type": "diagnosis_formula", "auto_extracted": False},
        {
            "rule_id": "auto_other",
            "rule_type": "diagnosis_formula",
            "auto_extracted": True,
            "source": {"source_path": "minzdrav_protocols/gastroenterologiya/other.pdf"},
        },
    ]
    matched = [{"source_path": "minzdrav_protocols/gastroenterologiya/КП_gerd.pdf"}]
    out = filter_rules_for_matched_protocols(rules, matched)
    assert any(r["rule_id"] == "manual" for r in out)
    assert not any(r["rule_id"] == "auto_other" for r in out)


def test_dedupe_single_diagnosis_formula():
    rules = [
        {"rule_id": "a", "rule_type": "diagnosis_formula", "auto_extracted": False},
        {"rule_id": "b", "rule_type": "diagnosis_formula", "auto_extracted": False},
    ]
    out = filter_rules_for_matched_protocols(rules, None)
    assert sum(1 for r in out if r.get("rule_type") == "diagnosis_formula") == 1


def test_loader_sees_new_conditions():
    clear_clinical_knowledge_cache()
    from clinical_knowledge.loader import load_conditions

    conds = load_conditions()
    assert "crohn" in conds
    assert "celiac" in conds
