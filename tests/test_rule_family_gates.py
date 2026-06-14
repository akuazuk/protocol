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
