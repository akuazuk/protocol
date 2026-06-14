"""Venous I80.x gates: no arterial/cardiology diagnosis_formula FP."""
from __future__ import annotations

from clinical_knowledge.consult_facts import extract_consult_facts_heuristic
from clinical_knowledge.rule_checker import run_rule_checker


def _failed_rule_ids(text: str) -> list[str]:
    facts = extract_consult_facts_heuristic(text)
    rc = run_rule_checker(facts, matched_protocols=[])
    return [
        str(f.get("rule_id") or "")
        for f in (rc.get("findings") or [])
        if not f.get("passed") and not f.get("skipped")
    ]


def test_pl_1_f_fixture_no_arterial_diagnosis_formula():
    text = """\
Врач: флеболог
Диагноз: I80.1 Флеботромбоз поверхностных вен нижней конечности.
Рекомендации: ривароксабан 20 мг.
"""
    ids = _failed_rule_ids(text)
    assert not any("529b5cfa" in r or "cardiology" in r for r in ids)
    assert not any("aacfc045" in r or "peripheral_artery" in r for r in ids)
    assert not any("aortic" in r for r in ids)


def test_pl_1_f_rich_deep_segment_dedupes_diagnosis_formula_messages():
    text = """\
Диагноз: I80.1 Флебит и тромбофлебит бедренной вены.
Флеботромбоз бедренно-подколенно-берцового сегмента правой нижней конечности.
"""
    facts = extract_consult_facts_heuristic(text)
    rc = run_rule_checker(facts, matched_protocols=[])
    failed = [
        f for f in (rc.get("findings") or [])
        if not f.get("passed") and not f.get("skipped") and f.get("rule_type") == "diagnosis_formula"
    ]
    messages = [str(f.get("message_ru") or "") for f in failed]
    assert len(messages) == len(set(messages))


def test_f1_p_rich_diagnosis_passes_dvt_formula():
    text = """\
Диагноз: I80.1. Флебит и тромбофлебит бедренной вены;
I80.2. Флебит и тромбофлебит других глубоких сосудов нижних конечностей;
ТГВ справа. Неоклюзионный тромбоз бедренной, оклюзионный тромбоз подколенной вены.
"""
    ids = _failed_rule_ids(text)
    assert not any("e912b455" in r for r in ids)
