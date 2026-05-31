"""Тесты safety_checker (ТЗ §17)."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.safety_checker import has_unhandled_critical, run_safety_checks

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "consultations"


def test_oncology_red_flag_detected():
    text = (FIXTURES / "surgery_redflag.txt").read_text(encoding="utf-8")
    doc = parse_consultation(text, consultation_id="onco")
    safety = run_safety_checks(doc)
    assert safety
    assert any(s.severity == "critical" for s in safety)


def test_good_kz_no_critical_flags():
    text = (FIXTURES / "gastro_adult.txt").read_text(encoding="utf-8")
    doc = parse_consultation(text, consultation_id="gastro")
    safety = run_safety_checks(doc)
    assert not any(s.severity == "critical" and s.status != "handled" for s in safety)
    assert not has_unhandled_critical(safety)


def test_handled_flag_when_routing_present():
    text = """\
Врач: хирург
Дата консультации: 01.01.2024
Объективный статус: образование кишки.
Диагноз: подозрение на опухоль.
Рекомендации по обследованию: направлен на консультацию онколога, колоноскопия.
"""
    doc = parse_consultation(text, consultation_id="handled")
    safety = run_safety_checks(doc)
    critical = [s for s in safety if s.severity == "critical"]
    if critical:
        assert all(s.status == "handled" for s in critical)
