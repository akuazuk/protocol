"""Тесты compliance engine, safety и scoring (ТЗ раздел 24)."""
from __future__ import annotations

from clinical_knowledge.compliance_engine import build_compliance_report
from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.consult_schema import ScoreBreakdown
from clinical_knowledge.safety_checker import run_safety_checks
from clinical_knowledge.scoring import compute_overall

GOOD_KZ = """\
Врач: гастроэнтеролог
Дата консультации: 14.07.2024
Дата рождения: 12.05.1976
Пол: мужской
Жалобы: боли в эпигастрии.
Анамнез: около 2 лет.
Объективный статус: удовлетворительное.
Диагноз: K29.7 Хронический гастрит.
Рекомендации по лечению: Омепразол 20 мг 2 раза в день 14 дней.
Дата повторной явки: 28.07.2024
"""

ONCO_KZ = """\
Врач: хирург
Дата консультации: 14.07.2024
Жалобы: боли в животе.
Объективный статус: образование в проекции кишки.
Диагноз: Нельзя исключить инвазию, образование кишки.
Рекомендации по лечению: симптоматически.
"""


def test_red_flag_forces_manual_review():
    doc = parse_consultation(ONCO_KZ, consultation_id="onco")
    safety = run_safety_checks(doc)
    assert any(s.severity == "critical" for s in safety)
    rep = build_compliance_report(doc, matches=[], rules_check={})
    assert rep.overall_status == "manual_review_required"
    assert rep.critical_issues


def test_suspected_diagnosis_assessed_as_suspected():
    doc = parse_consultation(ONCO_KZ, consultation_id="susp")
    rep = build_compliance_report(doc, matches=[], rules_check={})
    assert any(d.status == "suspected_needs_confirmation" for d in rep.diagnosis_assessments)


def test_required_exams_passed_not_missing():
    rules_check = {
        "findings": [
            {"rule_id": "r1", "rule_type": "required_exam", "exam": "ОАК", "passed": True,
             "source": {"source_path": "p.pdf"}},
            {"rule_id": "r2", "rule_type": "required_exam", "exam": "ФГДС", "passed": False,
             "source": {"source_path": "p.pdf"}},
        ]
    }
    doc = parse_consultation(GOOD_KZ, consultation_id="ex")
    rep = build_compliance_report(doc, matches=[], rules_check=rules_check)
    statuses = {e.exam_name: e.status for e in rep.exam_assessments}
    assert rep.score_breakdown.required_exams_score == 50.0
    assert "missing_required" in statuses.values()


def test_insufficient_data_when_too_few_blocks():
    bd = ScoreBreakdown(diagnosis_score=80.0)  # один блок
    overall, status = compute_overall(bd)
    assert overall is None
    assert status == "insufficient_data"


def test_applicable_match_scores_protocol_block():
    doc = parse_consultation(GOOD_KZ, consultation_id="good")
    matches = [{
        "protocol_id": "p1", "title": "КП гастрит", "source_path": "gastro/g.pdf",
        "population": "adult", "icd10_primary": ["K29.7"], "match_score": 80,
        "applicability": "applicable", "match_reasons": ["ok"], "mismatch_reasons": [],
    }]
    rep = build_compliance_report(doc, matches=matches, rules_check={})
    assert rep.score_breakdown.protocol_match_score == 90.0
    assert any(d.status == "supported" for d in rep.diagnosis_assessments)
    assert rep.overall_status in ("compliant", "mostly_compliant", "partially_compliant")
