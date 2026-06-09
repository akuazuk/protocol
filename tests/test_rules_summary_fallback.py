"""Тест fallback правил из Protocol Summary по МКБ."""
from __future__ import annotations

from clinical_knowledge.rules_summary_fallback import apply_summary_rules_fallback


def test_summary_fallback_skips_when_rules_already_positive(monkeypatch):
    monkeypatch.setenv("CONSULT_RULES_SUMMARY_FALLBACK", "1")
    clinical = {
        "consult_facts": {"consultation": {"icd10": ["K29.7"], "diagnosis_text": "гастрит"}},
        "matched_protocols": [],
        "rules_check": {"rules_compliance_pct": 42.0, "findings": [{"passed": True}]},
    }
    out = apply_summary_rules_fallback(clinical, ["K29.7"])
    assert out is clinical
    assert out["rules_check"]["rules_compliance_pct"] == 42.0


def test_summary_fallback_disabled(monkeypatch):
    monkeypatch.setenv("CONSULT_RULES_SUMMARY_FALLBACK", "0")
    clinical = {
        "consult_facts": {"consultation": {"icd10": ["K29.7"]}},
        "rules_check": {"rules_compliance_pct": 0, "findings": []},
    }
    out = apply_summary_rules_fallback(clinical, ["K29.7"])
    assert out is clinical
