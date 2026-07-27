"""Тесты risk-gate v3 и инвариантов применимости (Workstreams D/E ТЗ overnight-v1)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.kz_evaluation_engine import evaluate_kz_v3, gate_v3
from clinical_knowledge.kz_evaluation_schema import EvaluationFinding, KzEvaluationResultV3
from clinical_knowledge.kz_protocol_applicability import assess_applicability

_CASE = {
    "complaints": "боль",
    "anamnesis_doctor": "анамнез",
    "objective_status": "статус",
    "clinical_diagnosis": "Диагноз",
    "mkb_code_main": "J02.9",
    "treatment_recommendations": "парацетамол 500 мг",
}


def _result_with_finding(finding: EvaluationFinding) -> KzEvaluationResultV3:
    r = KzEvaluationResultV3(score_pct=80.0, status="good")
    r.findings.append(finding)
    return r


def test_cd_critical_does_not_block_gate():
    f = EvaluationFinding(
        code="C_red_flag_unrouted", axis="safety", severity="P0", kind="needs_human",
        passed=False, trust_level="C", penalty_applied=False, needs_human=True,
    )
    g = gate_v3(_result_with_finding(f))
    assert g["block"] is False


def test_ab_p0_blocks_gate():
    f = EvaluationFinding(
        code="D_reg55_p0", axis="regulatory", severity="P0", kind="regulatory_defect",
        passed=False, trust_level="A", penalty_applied=True,
    )
    g = gate_v3(_result_with_finding(f))
    assert g["block"] is True
    assert g["review_required"] is True


def test_low_confidence_requires_review_not_block():
    r = KzEvaluationResultV3(score_pct=82.0, status="good")
    r.confidence.overall = 0.3
    g = gate_v3(r)
    assert g["block"] is False
    assert g["review_required"] is True


def test_low_coverage_requires_review():
    r = KzEvaluationResultV3(score_pct=82.0, status="good")
    r.coverage.overall = 0.3
    g = gate_v3(r)
    assert g["review_required"] is True


def test_ab_p0_applies_hard_cap_in_engine():
    # reg55 P0 (trust A) должен ограничить overall и дать critical
    case = {**_CASE}
    r = evaluate_kz_v3(case)
    # если reg55 дал P0-дефект -> статус critical и cap
    p0 = [f for f in r.findings if f.severity == "P0" and f.penalty_applied]
    if p0:
        assert r.status == "critical"
        assert r.risk.cap_applied is True
        assert r.score_pct is not None and r.score_pct <= 40.0


def test_gate_disabled_by_default():
    r = evaluate_kz_v3(_CASE)
    assert r.mode.gate is False
    g = gate_v3(r)
    assert g["gate_enabled"] is False


# --- инварианты применимости (§9.2) ---
def test_child_protocol_not_penalizing_adult():
    case = {"patient_age_years": 40, "clinical_diagnosis": "x", "mkb_code_main": "J02"}
    proto = {"condition_id": "c", "name": "Острый фарингит у детей (детское население)", "match_score": 0.9}
    appl = assess_applicability(case, proto)
    assert appl is not None
    assert appl.population_match is False
    assert appl.penalty_eligible is False


def test_inpatient_protocol_not_penalizing_outpatient_kz():
    case = {"patient_age_years": 40, "clinical_diagnosis": "x"}
    proto = {"condition_id": "c", "name": "Лечение в стационарных условиях", "match_score": 0.9}
    appl = assess_applicability(case, proto)
    assert appl.care_setting_match is False
    assert appl.penalty_eligible is False


def test_fallback_protocol_advisory_only():
    case = {"patient_age_years": 40, "clinical_diagnosis": "x"}
    proto = {"condition_id": "c", "name": "Общий протокол", "is_fallback": True, "match_score": 0.9}
    appl = assess_applicability(case, proto)
    assert appl.penalty_eligible is False
    assert appl.applicability_confidence <= 0.3


def test_low_match_score_advisory():
    case = {"patient_age_years": 40, "clinical_diagnosis": "x"}
    proto = {"condition_id": "c", "name": "Профильный протокол", "match_score": 0.5}
    appl = assess_applicability(case, proto)
    assert appl.penalty_eligible is False


def test_good_match_adult_outpatient_penalty_eligible():
    case = {"patient_age_years": 40, "clinical_diagnosis": "x"}
    proto = {"condition_id": "c", "name": "Острый фарингит (взрослое население)", "match_score": 0.95}
    appl = assess_applicability(case, proto)
    assert appl.penalty_eligible is True
