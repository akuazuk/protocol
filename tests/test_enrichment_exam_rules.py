"""LLM-enrichment → required_exam rules."""
from __future__ import annotations

from clinical_knowledge.rules_from_enrichment import enrichment_payload_to_rules


def test_enrichment_required_exams_to_rules():
    payload = {
        "condition_id": "test_cond",
        "text_hash": "abcd1234",
        "enrichment": {
            "diagnosis_required_components": ["нозология"],
            "required_exams": ["ЭКГ", "ФГДС"],
            "red_flags": ["кровотечение"],
        },
    }
    rules = enrichment_payload_to_rules(payload)
    types = {r["rule_type"] for r in rules}
    assert "required_exam" in types
    assert "keyword_presence" in types
    exams = [r.get("exam") for r in rules if r["rule_type"] == "required_exam"]
    assert "ЭКГ" in exams
