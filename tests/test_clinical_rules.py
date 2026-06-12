"""Tests for clinical knowledge MVP (gastro rules)."""
from __future__ import annotations

from clinical_knowledge import (
    clinical_knowledge_status,
    extract_consult_facts_heuristic,
    match_protocol_cards,
    run_rule_checker,
)


def test_clinical_knowledge_status_has_gastro():
    st = clinical_knowledge_status()
    assert st.get("conditions", 0) >= 1
    assert st.get("rules", 0) >= 1


def test_extract_consult_facts_icd_and_gerd_hint():
    text = """
    Жалобы: изжога 8 месяцев, кислая отрыжка 3 раза в неделю.
    Диагноз: ГЭРБ, неэрозивная форма, лёгкая степень, фаза обострения, без осложнений. K21.9
    """
    facts = extract_consult_facts_heuristic(text, demographics_meta={"audience": "adult", "age_years": 45})
    assert any(c.startswith("K21") for c in (facts["consultation"].get("icd10") or []))
    assert "gerd" in (facts["consultation"].get("conditions_hint") or [])


def test_rule_checker_gerd_diagnosis_formula_missing():
    text = "Диагноз: ГЭРБ"
    facts = extract_consult_facts_heuristic(text, demographics_meta={"audience": "adult"})
    result = run_rule_checker(facts, condition_ids=["gerd"])
    failed = [f for f in result["findings"] if not f.get("passed")]
    types = {f.get("rule_type") for f in failed}
    assert "diagnosis_formula" in types
    formula = next(f for f in failed if f.get("rule_type") == "diagnosis_formula")
    assert formula.get("title_ru")
    assert "ГЭРБ" in formula["title_ru"] or "диагноз" in formula["title_ru"].lower()
    assert "gerd_diagnosis_formula" not in formula["title_ru"]


def test_rule_checker_gerd_full_diagnosis_passes_formula():
    text = (
        "Диагноз: Гастроэзофагеальная рефлюксная болезнь (ГЭРБ), неэрозивная форма, "
        "лёгкая степень, фаза обострения, без осложнений. K21.9"
    )
    facts = extract_consult_facts_heuristic(text, demographics_meta={"audience": "adult"})
    result = run_rule_checker(facts, condition_ids=["gerd"])
    formula = next(f for f in result["findings"] if f.get("rule_type") == "diagnosis_formula")
    assert formula.get("passed") is True


def test_population_mismatch_child_on_adult_protocol():
    text = "Диагноз: ГЭРБ K21"
    facts = extract_consult_facts_heuristic(text, demographics_meta={"audience": "child", "age_years": 10})
    result = run_rule_checker(facts, condition_ids=["gerd"])
    crit = [f for f in result["findings"] if f.get("severity") == "critical" and not f.get("passed")]
    assert crit


def test_match_protocol_cards_by_icd():
    facts = {
        "patient_context": {"adult_or_child": "adult"},
        "consultation": {"icd10": ["K21.9"], "conditions_hint": ["gerd"]},
    }
    matched = match_protocol_cards(facts, specialty_slug="gastroenterologiya", limit=3)
    assert matched
    top = matched[0]
    assert top.get("match_score", 0) > 0
    blob = ((top.get("title") or "") + " " + (top.get("source_path") or "")).lower()
    assert top.get("population") == "adult" or "пищевод" in blob or "желудк" in blob
