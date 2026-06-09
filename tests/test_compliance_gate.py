"""Тесты политики допуска КЗ к подписи (send_gate)."""
from clinical_knowledge.compliance_gate import evaluate_send_gate
from clinical_knowledge.consult_schema import (
    ComplianceIssue,
    ComplianceReport,
    SafetyAssessment,
    ScoreBreakdown,
)


def _report(score: float, *, critical: bool = False) -> ComplianceReport:
    r = ComplianceReport(
        consultation_id="t",
        overall_score=score,
        overall_status="mostly_compliant",
        score_breakdown=ScoreBreakdown(overall_score=score),
    )
    if critical:
        r.critical_issues.append(
            ComplianceIssue(
                issue_type="test",
                severity="critical",
                message_ru="тест",
            )
        )
    return r


def test_inform_always_allows():
    g = evaluate_send_gate(_report(40.0, critical=True), mode="inform")
    assert g["gate_allowed"] is True
    assert g["send_risk_level"] == "high"


def test_hard_gate_blocks_low_score():
    g = evaluate_send_gate(_report(50.0), mode="hard_gate", min_score_hard=70.0)
    assert g["gate_allowed"] is False
    assert g["send_risk_level"] == "blocked"


def test_critical_only_blocks_critical():
    g = evaluate_send_gate(_report(90.0, critical=True), mode="critical_only")
    assert g["gate_allowed"] is False


def test_soft_gate_requires_override_on_low_score():
    g = evaluate_send_gate(_report(60.0), mode="soft_gate", min_score_hard=70.0)
    assert g["gate_allowed"] is True
    assert g["requires_override"] is True


def test_gate_uses_min_of_headline_and_structural():
    g = evaluate_send_gate(_report(79.0), headline_score=50.0, mode="hard_gate", min_score_hard=70.0)
    assert g["gate_score"] == 50.0
    assert g["gate_allowed"] is False
    assert g["headline_score"] == 50.0
    assert g["structural_score"] == 79.0


def test_safety_critical_blocks_hard_gate():
    r = _report(95.0)
    r.safety_assessments.append(
        SafetyAssessment(
            issue_type="possible_malignancy",
            severity="critical",
            finding_text="x",
            status="not_handled",
        )
    )
    g = evaluate_send_gate(r, mode="hard_gate")
    assert g["gate_allowed"] is False
