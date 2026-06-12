"""Тесты детерминированного гибрида «Ориентировочное соответствие»."""
from __future__ import annotations

from clinical_knowledge.consult_overall_score import (
    SCORER_VERSION,
    apply_hybrid_overall_compliance,
)


def _structured(overall: float, *, status: str = "mostly_compliant") -> dict:
    return {
        "compliance": {
            "overall_score": overall,
            "overall_status": status,
            "score_breakdown": {"overall_score": overall, "diagnosis_score": overall},
        },
        "rules_check": {"rules_compliance_pct": 80.0},
    }


def test_hybrid_same_input_same_output() -> None:
    review = {
        "overall_compliance_pct": 12,
        "criteria": [
            {"name_ru": "Диагноз", "score_pct": 90},
            {"name_ru": "Лечение", "score_pct": 40},
        ],
    }
    sa = _structured(82.0)
    rules = {"rules_check": {"rules_compliance_pct": 70.0}}

    outs = [
        apply_hybrid_overall_compliance(
            dict(review),
            structured_analysis=sa,
            clinical_rules=rules,
        )["overall_compliance_pct"]
        for _ in range(5)
    ]
    assert len(set(outs)) == 1
    assert outs[0] == 80  # round(0.80*82 + 0.20*70)


def test_llm_criteria_do_not_change_hybrid_when_structured_present() -> None:
    sa = _structured(75.0)
    rules = {"rules_check": {"rules_compliance_pct": 60.0}}

    r1 = apply_hybrid_overall_compliance(
        {"criteria": [{"name_ru": "Диагноз", "score_pct": 95}]},
        structured_analysis=sa,
        clinical_rules=rules,
    )
    r2 = apply_hybrid_overall_compliance(
        {"criteria": [{"name_ru": "Диагноз", "score_pct": 20}]},
        structured_analysis=sa,
        clinical_rules=rules,
    )
    assert r1["overall_compliance_pct"] == r2["overall_compliance_pct"] == 72


def test_fallback_weighted_llm_without_structured() -> None:
    review = {
        "criteria": [
            {"name_ru": "Соответствие диагноза", "score_pct": 80},
            {"name_ru": "Лечение и назначения", "score_pct": 60},
        ],
    }
    rules = {"rules_check": {"rules_compliance_pct": 50.0}}
    out = apply_hybrid_overall_compliance(review, structured_analysis=None, clinical_rules=rules)
    assert out["overall_compliance_method"].startswith("hybrid_llm_")
    assert 50 <= out["overall_compliance_pct"] <= 80


def test_safety_cap_lowers_score() -> None:
    sa = _structured(88.0, status="manual_review_required")
    out = apply_hybrid_overall_compliance(
        {"criteria": []},
        structured_analysis=sa,
        clinical_rules={"rules_check": {"rules_compliance_pct": 90.0}},
    )
    assert out["overall_compliance_pct"] <= 45
    assert "_safety_cap" in out["overall_compliance_method"]


def test_scorer_version_attached() -> None:
    out = apply_hybrid_overall_compliance(
        {"criteria": []},
        structured_analysis=_structured(70.0),
        clinical_rules=None,
    )
    assert out["overall_compliance_scorer_version"] == SCORER_VERSION


def test_rules_zero_without_matched_protocols_not_blended() -> None:
    sa = _structured(78.0)
    rules = {"rules_check": {"rules_compliance_pct": 0.0}, "matched_protocols": []}
    out = apply_hybrid_overall_compliance(
        {"criteria": []},
        structured_analysis=sa,
        clinical_rules=rules,
    )
    assert out["overall_compliance_pct"] == 78
    assert out["overall_compliance_components"]["rules"] is None


def test_partially_handled_thrombosis_no_flat_fifty_cap() -> None:
    """Частично учтённый тромбоз не должен обрезать гибрид ровно до 50%."""
    sa = {
        "compliance": {
            "overall_score": 79.0,
            "overall_status": "mostly_compliant",
            "safety_assessments": [
                {
                    "issue_type": "thrombosis",
                    "severity": "high",
                    "status": "partially_handled",
                    "finding_text": "флеботромбоз",
                }
            ],
            "critical_issues": [],
            "safety_cap": {"applied": False},
        }
    }
    rules = {"rules_check": {"rules_compliance_pct": 29.0}, "matched_protocols": [{"protocol_id": "p1"}]}
    out = apply_hybrid_overall_compliance(
        {"criteria": []},
        structured_analysis=sa,
        clinical_rules=rules,
    )
    assert out["overall_compliance_pct"] == 69  # round(0.80*79 + 0.20*29)


def test_rules_zero_warnings_only_not_blended() -> None:
    sa = _structured(74.0)
    rules = {
        "rules_check": {
            "rules_compliance_pct": 0.0,
            "findings": [
                {"passed": False, "severity": "warning", "rule_type": "diagnosis_formula"},
            ],
        },
        "matched_protocols": [{"protocol_id": "p1"}],
    }
    out = apply_hybrid_overall_compliance(
        {"criteria": []},
        structured_analysis=sa,
        clinical_rules=rules,
    )
    assert out["overall_compliance_pct"] == 74
    assert out["overall_compliance_components"]["rules"] is None
    assert "_safety_cap" not in (out.get("overall_compliance_method") or "")
