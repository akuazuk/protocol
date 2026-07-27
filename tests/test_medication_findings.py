"""Тесты trust-aware находок по терапии (Workstream I ТЗ overnight-v1)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.medication_findings import classify_medication_findings


def _codes(findings):
    return {f.code for f in findings}


def test_double_nsaid_is_safety_warning():
    case = {"treatment_recommendations": "ибупрофен 400 мг 3 раза и диклофенак 50 мг 2 раза"}
    f = classify_medication_findings(case)
    assert "MED_nsaid_dup" in _codes(f)
    nsaid = [x for x in f if x.code == "MED_nsaid_dup"][0]
    assert nsaid.kind == "safety_warning"
    assert nsaid.penalty_applied is True


def test_missing_dose_is_documentation_gap():
    case = {"treatment_recommendations": "амоксициллин внутрь"}
    f = classify_medication_findings(case)
    codes = _codes(f)
    # либо распознан препарат без дозы -> documentation_gap
    if "MED_missing_dose" in codes:
        gap = [x for x in f if x.code == "MED_missing_dose"][0]
        assert gap.kind == "documentation_gap"
        assert gap.penalty_applied is True


def test_dose_context_missing_not_penalized():
    # доза есть, но нет возраста/массы -> insufficient_context, БЕЗ штрафа
    case = {"treatment_recommendations": "эноксапарин 40 мг подкожно"}
    f = classify_medication_findings(case)
    ctx = [x for x in f if x.code == "MED_dose_context_missing"]
    if ctx:
        assert ctx[0].kind == "insufficient_context"
        assert ctx[0].penalty_applied is False
        assert ctx[0].needs_human is True


def test_empty_treatment_no_findings():
    assert classify_medication_findings({}) == []
