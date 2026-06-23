"""Tests for protocol_match_ui."""

from clinical_knowledge.protocol_match_ui import (
    compute_match_explain_ru,
    compute_match_tier,
    enrich_protocol_match_ui,
)


def test_icd_primary_tier():
    pr = {"matched_icd_codes": ["J02.9"], "icd_match_strength": 85.0, "confidence_score": 0.9}
    assert compute_match_tier(pr, ["J02.9"]) == "icd_primary"


def test_text_only_tier():
    pr = {"confidence_score": 0.7}
    assert compute_match_tier(pr, []) == "text_only"


def test_manual_check_tier():
    pr = {"confidence_score": 0.3}
    assert compute_match_tier(pr, []) == "manual_check"


def test_enrich_adds_fields():
    rows = enrich_protocol_match_ui(
        [{"path": "a.pdf", "matched_icd_codes": ["J02.9"], "icd_match_strength": 80}],
        ["J02.9"],
    )
    assert rows[0]["match_tier"] == "icd_primary"
    assert rows[0]["match_explain_ru"]


def test_explain_ru_icd_primary():
    pr = {"match_tier": "icd_primary", "matched_icd_codes": ["J04.0"]}
    text = compute_match_explain_ru(pr, ["J02.9"])
    assert "J02.9" in text or "J04.0" in text
