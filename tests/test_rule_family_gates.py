"""Rule family gates: top override rule_id patterns from Methodist feedback."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.consult_facts import extract_consult_facts_heuristic
from clinical_knowledge.consult_retrieval import consult_target_protocol_paths
from clinical_knowledge.rule_checker import run_rule_checker
from clinical_knowledge.rule_family_gates import is_oncology_icd

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "consultations"


def _failed_rule_ids(text: str) -> list[str]:
    facts = extract_consult_facts_heuristic(text)
    rc = run_rule_checker(facts, matched_protocols=[])
    return [
        str(f.get("rule_id") or "")
        for f in (rc.get("findings") or [])
        if not f.get("passed") and not f.get("skipped")
    ]


def test_d25_benign_myoma_not_oncology_icd():
    assert not is_oncology_icd("D25.2")
    assert is_oncology_icd("C50.9")
    assert is_oncology_icd("D37.9")
    assert not is_oncology_icd("D21.0")


def test_obgyn_61_no_neoplasm_diagnosis_formula():
    text = (FIXTURES / "feedback_obgyn_61.txt").read_text(encoding="utf-8")
    ids = _failed_rule_ids(text)
    assert not any("9bdafb96" in r or "neoplasm" in r for r in ids)


def test_urvi_j06_no_bronchitis_diagnosis_formula():
    text = (FIXTURES / "feedback_urvi_j06.txt").read_text(encoding="utf-8")
    ids = _failed_rule_ids(text)
    assert not any("acute_bronchitis" in r or "e6302486" in r for r in ids)


def test_thyroid_euti_no_obesity_formula():
    text = (FIXTURES / "feedback_thyroid_eu.txt").read_text(encoding="utf-8")
    ids = _failed_rule_ids(text)
    assert not any("obesity" in r or "4c60ee77" in r for r in ids)


def test_j06_expanded_slugs_find_urti_protocol():
    paths, meta = consult_target_protocol_paths(
        merged_icd=["J06.9"],
        diag_icd=["J06.9"],
        clinical_rules=None,
        specialty_slugs=["terapiya"],
    )
    assert paths, meta
    assert "pulmonologiya-ftiziatriya" in meta.get("specialty_slugs") or "infektsionnye-zabolevaniya" in meta.get(
        "specialty_slugs"
    )
    assert any("респиратор" in p.lower() or "орви" in p.lower() or "вирус" in p.lower() for p in paths)


def test_kard_1_e55_no_thyroid_diagnosis_formula():
    from clinical_knowledge.text_extract import extract_text_from_path

    pdf = FIXTURES.parents[1] / "clients_consult" / "kard_1.pdf"
    if not pdf.is_file():
        return
    text = extract_text_from_path(pdf)
    ids = _failed_rule_ids(text)
    assert not any("thyroid" in r or "2d00ba03" in r for r in ids)


def test_gastro_1_oncology_suspicion_no_dyspepsia_formula():
    text = (FIXTURES / "gastro_1.txt").read_text(encoding="utf-8")
    ids = _failed_rule_ids(text)
    assert not any("8e7327d9" in r or "functional_dyspepsia" in r for r in ids)


def test_gastro_1_oncology_suspicion_caps_overall():
    from clinical_knowledge.consult_analysis import analyze_consultation_text

    text = (FIXTURES / "gastro_1.txt").read_text(encoding="utf-8")
    out = analyze_consultation_text(text, consultation_id="gastro_1", with_markdown=False)
    comp = out.get("compliance") or {}
    overall = comp.get("overall_score")
    assert overall is not None and overall <= 75.0


def test_gastro_heuristic_facts_multiline_diagnosis_oncology():
    text = (FIXTURES / "gastro_1.txt").read_text(encoding="utf-8")
    from clinical_knowledge.consult_facts import extract_consult_facts_heuristic
    from clinical_knowledge.rule_family_gates import has_oncology_clinical_suspicion

    facts = extract_consult_facts_heuristic(text)
    assert "опухол" in (facts.get("consultation") or {}).get("diagnosis_text", "").lower()
    assert has_oncology_clinical_suspicion(facts)
    ids = _failed_rule_ids(text)
    assert not any("8e7327d9" in r or "functional_dyspepsia" in r for r in ids)


def test_report_g_11_anamnesis_pregnancy_count_not_active_pregnancy():
    text = (FIXTURES / "report_g_11_anamnesis.txt").read_text(encoding="utf-8")
    ids = _failed_rule_ids(text)
    assert not any("d4c0214b" in r or "pregnancy" in r for r in ids)


def test_f1_p_rich_safety_and_follow_scores():
    from clinical_knowledge.consult_analysis import analyze_consultation_text

    text = (FIXTURES / "f1_p_rich.txt").read_text(encoding="utf-8")
    out = analyze_consultation_text(text, consultation_id="f1_p_rich", with_markdown=False)
    comp = out.get("compliance") or {}
    bd = comp.get("score_breakdown") or {}
    assert (bd.get("safety_score") or 0) >= 75.0
    assert (bd.get("follow_up_score") or 0) >= 85.0


def _pdf_text(name: str) -> str | None:
    from clinical_knowledge.text_extract import extract_text_from_path

    pdf = FIXTURES.parents[1] / "clients_consult" / name
    if not pdf.is_file():
        return None
    return extract_text_from_path(pdf)


def test_report_ter_1_no_gastritis_thyroid_trauma_rules():
    text = _pdf_text("report_ter_1.pdf")
    if not text:
        return
    ids = _failed_rule_ids(text)
    assert not any(
        x in r
        for r in ids
        for x in ("gastritis", "abdominal_trauma", "thyroid", "2d00ba03", "9f9e0fb1", "5e8308e8")
    )


def test_report_lor_1_no_bronchitis_formula_and_better_safety():
    from clinical_knowledge.consult_analysis import analyze_consultation_text

    text = _pdf_text("report_lor_1.pdf")
    if not text:
        return
    ids = _failed_rule_ids(text)
    assert not any("acute_bronchitis" in r or "e6302486" in r for r in ids)
    out = analyze_consultation_text(text, consultation_id="report_lor_1", with_markdown=False)
    bd = (out.get("compliance") or {}).get("score_breakdown") or {}
    assert (bd.get("safety_score") or 0) >= 75.0


def test_report_urolofg_2_no_thyroid_formula():
    text = _pdf_text("report_urolofg_2.pdf")
    if not text:
        return
    ids = _failed_rule_ids(text)
    assert not any("thyroid" in r or "2d00ba03" in r for r in ids)
