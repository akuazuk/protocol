from __future__ import annotations

import pytest

from clinical_knowledge.mo_dx_evidence_score import (
    dx_evidence_eligibility,
    nonsemantic_dx_result,
    validate_dx_evidence_result,
)
from clinical_knowledge.mo_plan_protocol_score import (
    resolve_plan_route,
    selected_plan_score,
    validate_plan_concordance_result,
)


def test_dx_contract_accepts_evidence_grounded_semantic_result() -> None:
    result = validate_dx_evidence_result(
        {
            "case_id": "case-1",
            "dx_evidence_pct": 35,
            "verdict": "poor",
            "supported_by": [{"slot": "complaints", "evidence": "один поддерживающий симптом"}],
            "not_supported_by": [{"slot": "exam_data", "evidence": "критерий не подтверждён"}],
            "contradictions": [{"slot": "objective_status", "evidence": "противоречащая находка"}],
            "icd_fit": "mismatch",
            "potential_harm": True,
            "provenance": "llm_blind",
        }
    )
    assert result["dx_evidence_pct"] == 35
    assert result["verdict"] == "poor"
    assert result["not_supported_by"][0]["slot"] == "exam_data"
    assert result["potential_harm"] is True


def test_dx_contract_rejects_poor_without_evidence() -> None:
    with pytest.raises(ValueError, match="requires"):
        validate_dx_evidence_result(
            {
                "dx_evidence_pct": 20,
                "verdict": "poor",
                "icd_fit": "unknown",
                "provenance": "llm_blind",
            }
        )


def test_dx_without_icd_is_eligible_and_not_penalized() -> None:
    eligibility = dx_evidence_eligibility(
        {
            "slots": {
                "clinical_diagnosis": "Содержательный диагноз",
                "complaints": "Жалобы присутствуют",
            }
        }
    )
    assert eligibility["status"] == "eligible"
    assert eligibility["icd_present"] is False
    assert eligibility["icd_absence_penalty"] is False
    assert nonsemantic_dx_result(
        {"slots": {"clinical_diagnosis": "Диагноз", "complaints": "Жалобы"}}
    ) is None


def test_dx_missing_diagnosis_is_na_and_missing_evidence_is_blocked() -> None:
    missing_dx = nonsemantic_dx_result({"slots": {"complaints": "Жалобы"}})
    assert missing_dx is not None
    assert missing_dx["verdict"] == "na"
    blocked = nonsemantic_dx_result({"slots": {"clinical_diagnosis": "Диагноз"}})
    assert blocked is not None
    assert blocked["verdict"] == "blocked"
    assert blocked["dx_evidence_pct"] is None
    lab_only = nonsemantic_dx_result(
        {
            "slots": {
                "clinical_diagnosis": "Диагноз",
                "lab": "ОАК (2026-08-20, день визита)",
            }
        }
    )
    assert lab_only is None


def test_grounded_plan_contract_keeps_three_blocks_and_sources() -> None:
    result = validate_plan_concordance_result(
        {
            "case_id": "case-1",
            "exam_protocol_pct": 80,
            "treatment_protocol_pct": 70,
            "followup_protocol_pct": 60,
            "plan_protocol_pct": 70,
            "verdict": "partial",
            "kp_status": "matched",
            "kp_path": "protocols/example.md",
            "kp_trust": "A",
            "missing_required": ["контроль"],
            "off_protocol": [],
            "source_refs": ["protocols/example.md#follow-up"],
            "provenance": "kp_grounded",
        }
    )
    assert selected_plan_score(result) == 70
    assert result["plan_general_llm_pct"] is None
    assert result["source_refs"]


def test_no_kp_fallback_never_claims_protocol_compliance() -> None:
    result = validate_plan_concordance_result(
        {
            "case_id": "case-2",
            "plan_general_llm_pct": 55,
            "verdict": "partial",
            "kp_status": "unmatched",
            "missing_required": ["наблюдение"],
            "off_protocol": [],
            "source_refs": [],
            "provenance": "llm_no_kp",
        }
    )
    assert selected_plan_score(result) == 55
    assert result["plan_protocol_pct"] is None
    assert result["kp_path"] == ""
    with pytest.raises(ValueError, match="cannot claim"):
        validate_plan_concordance_result(
            {
                "plan_general_llm_pct": 55,
                "plan_protocol_pct": 55,
                "verdict": "partial",
                "kp_status": "unmatched",
                "source_refs": ["fake"],
                "provenance": "llm_no_kp",
            }
        )


def test_plan_route_uses_clinical_hit_and_trust_threshold() -> None:
    grounded = resolve_plan_route(
        {
            "items": [
                {
                    "match_kind": "clinical",
                    "score": 75,
                    "trust": "A",
                    "path": "protocols/example.md",
                }
            ]
        }
    )
    assert grounded["route"] == "kp_grounded"
    fallback = resolve_plan_route(
        {
            "items": [
                {
                    "match_kind": "clinical",
                    "score": 75,
                    "trust": "C",
                    "path": "protocols/weak.md",
                }
            ]
        }
    )
    assert fallback["route"] == "llm_no_kp"
    assert fallback["fallback_reason"] == "kp_trust_below_threshold"
