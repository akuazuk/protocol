"""Tests for medication safety checks (dual NSAID)."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.consult_analysis import analyze_consultation_text, facts_from_document
from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.medication_safety import detect_concurrent_nsaids
from clinical_knowledge.rule_checker import run_rule_checker
from clinical_knowledge.safety_checker import run_safety_checks

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "consultations"


def test_detect_concurrent_nsaids_aertal_dexalgin():
    text = (FIXTURES / "feedback_dual_nsaid.txt").read_text(encoding="utf-8")
    doc = parse_consultation(text, consultation_id="dual_nsaid")
    hit = detect_concurrent_nsaids(doc)
    assert hit is not None
    assert hit.severity == "critical"
    assert "нпвп" in hit.finding_text.lower()


def test_dual_nsaid_lowers_treatment_and_safety_scores():
    text = (FIXTURES / "feedback_dual_nsaid.txt").read_text(encoding="utf-8")
    out = analyze_consultation_text(text, consultation_id="dual_nsaid", with_markdown=False)
    bd = out["compliance"]["score_breakdown"]
    assert bd["treatment_score"] is not None and bd["treatment_score"] <= 25.0
    assert bd["safety_score"] is not None and bd["safety_score"] <= 15.0
    safety = run_safety_checks(parse_consultation(text))
    assert any("нпвп" in (s.finding_text or "").lower() for s in safety)


def test_pregnancy_rule_skipped_for_61yo_obgyn_without_pregnancy():
    text = (FIXTURES / "feedback_obgyn_61.txt").read_text(encoding="utf-8")
    doc = parse_consultation(text, consultation_id="obgyn_61")
    facts = facts_from_document(doc)
    preg_path = (
        "minzdrav_protocols/akusherstvo-ginekologiya/"
        "КП_медицинское_наблюдение_и_оказание_медицинской_помощи_"
        "женщинам_в_акушерстве_и_гинекологии_пост_МЗ_19.02.2018_17.pdf"
    )
    rc = run_rule_checker(
        facts,
        matched_protocols=[{"source_path": preg_path, "applicability": "applicable"}],
    )
    preg = [f for f in rc.get("findings", []) if "d4c0214b" in str(f.get("rule_id", ""))]
    assert not preg


def _failed_ids(text: str, cid: str, *, protocols: list[dict] | None = None) -> set[str]:
    facts = facts_from_document(parse_consultation(text, consultation_id=cid))
    rc = run_rule_checker(facts, matched_protocols=protocols or [])
    return {str(f.get("rule_id") or "") for f in rc.get("findings", []) if f.get("passed") is False}


def test_obgyn_61_local_reanalyze_no_pregnancy_rule():
    text = (FIXTURES / "feedback_obgyn_61.txt").read_text(encoding="utf-8")
    out = analyze_consultation_text(text, consultation_id="obgyn_61_r109", with_markdown=False)
    failed = {
        str(i.get("issue_type") or i.get("rule_id") or "")
        for i in (out["compliance"].get("issues") or []) + (out["compliance"].get("warnings") or [])
    }
    assert not any("d4c0214b" in r or "pregnancy" in r for r in failed)


def test_urvi_j06_no_pneumonia_asthma_tb_rules():
    text = (FIXTURES / "feedback_urvi_j06.txt").read_text(encoding="utf-8")
    failed = _failed_ids(text, "urvi_j06")
    assert not any(x in failed for x in (
        "f3927f15_path_tuberculosis_diagnosis_formula",
        "49c2e461_auto_pneumonia_generic_diagnosis_formula",
        "c7955406_auto_bronchial_asthma_generic_diagnosis_formula",
    ))


def test_thyroid_euthyroid_no_obesity_rule():
    text = (FIXTURES / "feedback_thyroid_eu.txt").read_text(encoding="utf-8")
    failed = _failed_ids(text, "thyroid_eu")
    assert "4c60ee77_path_obesity_diagnosis_formula" not in failed
    assert "2d00ba03_path_thyroid_disease_diagnosis_formula" not in failed


def test_gastro_adult_no_pediatric_surgery_rule():
    from tests.test_regression_kz_compliance import GASTRO_1

    failed = _failed_ids(GASTRO_1, "gastro_oncology_adult")
    assert "7265d120_path_pediatric_general_surgery_diagnosis_formula" not in failed
