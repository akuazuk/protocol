"""Тесты requirement_checker (ТЗ §9-10)."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.requirement_checker import load_kz_requirements, run_requirement_check

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "consultations"


def test_load_kz_requirements_has_required_tiers():
    cfg = load_kz_requirements()
    assert isinstance(cfg.get("required"), list)
    assert len(cfg["required"]) >= 5
    assert "conditional" in cfg
    assert "recommended" in cfg


def test_good_kz_passes_required_sections():
    text = (FIXTURES / "gastro_adult.txt").read_text(encoding="utf-8")
    doc = parse_consultation(text, consultation_id="gastro")
    assessment, issues = run_requirement_check(doc)
    assert "diagnosis" in assessment.filled_sections
    assert "objective_status" in assessment.filled_sections
    assert assessment.structural_score >= 50.0
    assert assessment.patient_data_score >= 66.0


def test_redflag_kz_flags_missing_routing():
    text = (FIXTURES / "surgery_redflag.txt").read_text(encoding="utf-8")
    doc = parse_consultation(text, consultation_id="rf")
    assessment, issues = run_requirement_check(doc)
    critical_types = {i.issue_type for i in issues if i.severity == "critical"}
    assert "routing_red_flag" in critical_types or assessment.missing_conditional
