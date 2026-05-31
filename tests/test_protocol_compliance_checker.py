"""Тесты protocol_compliance_checker (ТЗ §12)."""
from __future__ import annotations

from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.protocol_compliance_checker import (
    enhance_treatment_assessments,
    findings_to_issues,
    run_protocol_compliance_check,
)

KZ = """\
Врач: гастроэнтеролог
Дата консультации: 14.07.2024
Диагноз: K29.7 Хронический гастрит.
Рекомендации по лечению: симптоматически.
"""


def test_findings_to_issues_with_source():
    doc = parse_consultation(KZ, consultation_id="t1")
    rules = {
        "findings": [{
            "rule_id": "r1",
            "rule_type": "required_exam",
            "passed": False,
            "severity": "warning",
            "message_ru": "Нет ФГДС",
            "source": {"source_path": "gastro/kp.pdf", "protocol_id": "p1"},
        }]
    }
    issues = findings_to_issues(rules, doc)
    assert len(issues) == 1
    assert issues[0].category == "required_exams"
    assert issues[0].source_refs


def test_enhance_treatment_from_keyword_rules():
    doc = parse_consultation(KZ, consultation_id="t2")
    rules = {
        "findings": [{
            "rule_id": "kw1",
            "rule_type": "keyword_presence",
            "keyword": "омепразол",
            "passed": False,
            "severity": "warning",
            "message_ru": "Ожидался ингибитор протонной помпы",
        }]
    }
    _, score = enhance_treatment_assessments(doc, rules, [])
    assert score is not None
    assert score < 100


def test_run_protocol_compliance_check():
    doc = parse_consultation(KZ, consultation_id="t3")
    issues, treatments, score = run_protocol_compliance_check(doc, {"findings": []}, [])
    assert isinstance(issues, list)
    assert score is None
