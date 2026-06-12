"""Tests for Methodist tier enrichment (L0/L1 clinical rules + tier meta)."""
from __future__ import annotations

from clinical_knowledge.methodist_enrich import (
    TIER_META,
    enrich_methodist_tier_payload,
)


def test_tier_meta_has_all_levels():
    assert set(TIER_META) == {"L0", "L1", "L2"}
    assert TIER_META["L0"]["latency_hint_ru"]
    assert TIER_META["L2"]["checks_ru"]


def test_enrich_l0_adds_tier_meta_and_rules(monkeypatch):
    monkeypatch.setenv("CONSULT_RULE_CHECK", "1")
    result = {
        "ok": True,
        "send_gate": {"gate_score": 72, "gate_allowed": True},
        "structured_analysis": {
            "compliance": {"overall_score": 72, "overall_status": "mostly_compliant"},
            "matches": [],
        },
    }
    out = enrich_methodist_tier_payload(
        result,
        tier="L0",
        full_text="Диагноз: K21.9 ГЭРБ. Рекомендована ФГДС.",
        category_slugs="gastroenterologiya",
        latency_ms=850,
    )
    assert out["review_tier"] == "L0"
    meta = out["methodist_tier_meta"]
    assert meta["label_ru"].startswith("L0")
    assert meta["latency_ms"] == 850
    assert "clinical_rules" in out
    assert out["clinical_rules"].get("rules_check") is not None


def test_enrich_l2_keeps_existing_clinical_rules():
    cr = {"rules_check": {"rules_compliance_pct": 90.0, "findings": []}}
    result = {"review_tier": "L2", "clinical_rules": cr, "retrieval_paths": ["a/b.pdf"]}
    out = enrich_methodist_tier_payload(
        result,
        tier="L2",
        full_text="Тест",
        latency_ms=120000,
    )
    assert out["clinical_rules"] is cr
    assert out["methodist_tier_meta"]["latency_label_ru"] == "120.0 с"
