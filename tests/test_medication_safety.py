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
