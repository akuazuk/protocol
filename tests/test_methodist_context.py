"""Tests for Methodist review context (compliance focus, no signature)."""
from __future__ import annotations

from clinical_knowledge.methodist_context import build_methodist_review_context
from clinical_knowledge.privacy import redact_kz_text_for_display


def test_redact_kz_text_hides_patient_name():
    raw = "Ф.И.О: Кузавка Павел Леонидович\nДиагноз: K21.9 ГЭРБ\nРекомендации: омепразол"
    out = redact_kz_text_for_display(raw)
    assert "Кузавка" not in out
    assert "Павел" not in out
    assert "K21.9" in out


def test_build_methodist_review_context_compliance_focus():
    result = {
        "review_tier": "L1",
        "review": {
            "overall_compliance_pct": 72,
            "overall_compliance_components": {"structured": 70, "rules": 80},
            "criteria": [],
        },
        "structured_analysis": {
            "compliance": {
                "overall_status": "mostly_compliant",
                "overall_score": 70,
                "score_breakdown": {
                    "diagnosis_score": 65.0,
                    "treatment_score": 80.0,
                },
                "critical_issues": [],
            }
        },
        "clinical_rules": {
            "rules_check": {
                "rules_compliance_pct": 80.0,
                "findings": [
                    {
                        "rule_id": "req_exam",
                        "passed": False,
                        "title_ru": "Нужна ФГДС",
                        "message_ru": "Не указана ФГДС",
                    }
                ],
            },
            "matched_protocols": [{"path": "gastro/kp.pdf"}],
        },
    }
    ctx = build_methodist_review_context(result, "Ф.И.О: Иванов Иван Иванович\nЖалобы: изжога")
    assert ctx["focus"] == "protocol_compliance"
    assert "send_gate" in ctx["exclude_from_review"]
    assert "Иванов" not in ctx["kz_text_display"]
    assert ctx["compliance"]["overall_pct"] == 72
    assert len(ctx["compliance"]["blocks"]) == 8
    assert ctx["rules_findings"][0]["rule_id"] == "req_exam"
