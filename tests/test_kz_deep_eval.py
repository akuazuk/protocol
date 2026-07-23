"""Тесты глубокой оценки КЗ: детекторы осей A/B/C + risk-gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.kz_deep_eval import _apply_risk_gate, evaluate_kz_deep, icd_validate


def _drug_ctx():
    ctx = {}
    ha = ROOT / "data" / "drug_safety" / "high_alert.json"
    st = ROOT / "data" / "drug_safety" / "stopp_start_beers.json"
    if ha.is_file():
        ctx["high_alert"] = json.loads(ha.read_text(encoding="utf-8"))
    if st.is_file():
        ctx["stopp"] = json.loads(st.read_text(encoding="utf-8"))
    # синтетический DDInter (детерминированно)
    ctx["ddinter"] = {"pairs": {"aspirin||warfarin": "Major"}}
    return ctx


def test_documentation_missing_blocks():
    case = {"complaints": "", "clinical_diagnosis": "J06.9 ОРВИ"}
    r = evaluate_kz_deep(case)
    codes = {f["code"] for f in r["findings"]}
    assert "A_missing_complaints" in codes
    assert r["axes"]["documentation"] < 100


def test_risk_gate_min_axis_rule():
    # Высокое среднее (мягкий дефолт -> good), но слабая ось concordance=45.
    axes = {"documentation": 95, "clinical_concordance": 45, "safety": 100, "regulatory": 90}
    findings = []  # без P0/P1
    ovr = round(sum(axes.values()) / 4, 1)  # 82.5
    # без min-axis правила: good
    _, st_default = _apply_risk_gate(ovr, findings, axes=axes, cfg={"t_good": 80, "t_acc": 60, "min_axis_review": None})
    assert st_default == "good"
    # с min-axis=55: слабая ось не маскируется -> review
    _, st_cal = _apply_risk_gate(ovr, findings, axes=axes, cfg={"t_good": 80, "t_acc": 60, "min_axis_review": 55})
    assert st_cal == "review"


def test_risk_gate_p0_overrides_min_axis():
    axes = {"documentation": 95, "clinical_concordance": 45, "safety": 100, "regulatory": 90}
    findings = [{"severity": "P0", "passed": False}]
    _, st = _apply_risk_gate(85.0, findings, axes=axes, cfg={"t_good": 80, "t_acc": 60, "min_axis_review": 55})
    assert st == "critical"


def test_icd_validate_format():
    assert icd_validate("J06.9", "ОРВИ")[0] is True
    assert icd_validate("XYZ", "нечто")[0] is False
    assert icd_validate("", "нет")[0] is False


def test_concordance_exams_gap():
    case = {
        "complaints": "кашель, температура",
        "anamnesis_doctor": "болен 3 дня",
        "objective_status": "в лёгких хрипы",
        "clinical_diagnosis": "Пневмония",
        "mkb_code_main": "J18.9",
        "exam_recommendations": "общий анализ крови",
        "treatment_recommendations": "амоксициллин 500 мг 3 раза",
    }
    protocol = {
        "required_exams": ["общий анализ крови", "рентгенография органов грудной клетки"],
        "treatment": ["антибактериальные средства", "амоксициллин"],
        "red_flags": [],
    }
    r = evaluate_kz_deep(case, protocol_ctx=protocol, drug_ctx=_drug_ctx())
    codes = {f["code"] for f in r["findings"]}
    assert "B_exams_gap" in codes  # рентген не отражён
    assert r["axes"]["clinical_concordance"] is not None


def test_safety_red_flag_p0_gate():
    case = {
        "complaints": "внезапная резкая головная боль, worst headache",
        "objective_status": "менингеальные знаки",
        "clinical_diagnosis": "Цефалгия",
        "treatment_recommendations": "",  # нет маршрутизации
        "dispensary_info": "",
        "return_date": "",
    }
    protocol = {"red_flags": ["менингеальные знаки"], "required_exams": [], "treatment": []}
    r = evaluate_kz_deep(case, protocol_ctx=protocol, drug_ctx=_drug_ctx())
    assert r["has_potential_harm"] is True
    assert r["overall_status"] == "critical"
    assert r["overall_pct"] <= 40


def test_safety_ddi_major():
    case = {
        "complaints": "боль", "clinical_diagnosis": "I48 Фибрилляция предсердий",
        "treatment_recommendations": "варфарин 5 мг; аспирин 100 мг",
        "dispensary_info": "явка через 7 дней",
    }
    r = evaluate_kz_deep(case, drug_ctx=_drug_ctx())
    codes = {f["code"] for f in r["findings"]}
    assert "C_ddi" in codes


def test_safety_high_alert_no_dose():
    case = {
        "complaints": "боль", "clinical_diagnosis": "I80 Тромбоз",
        "treatment_recommendations": "варфарин по схеме",  # без дозы
        "dispensary_info": "контроль МНО",
    }
    r = evaluate_kz_deep(case, drug_ctx=_drug_ctx())
    codes = {f["code"] for f in r["findings"]}
    assert "C_high_alert_no_dose" in codes


def test_safety_stopp_elderly_nsaid():
    case = {
        "complaints": "боль в суставах", "clinical_diagnosis": "M17 Гонартроз",
        "treatment_recommendations": "диклофенак 50 мг 2 раза",
        "patient_age_years": 74,
        "dispensary_info": "явка через 14 дней",
    }
    r = evaluate_kz_deep(case, drug_ctx=_drug_ctx())
    codes = {f["code"] for f in r["findings"]}
    assert any(c.startswith("C_STOPP") or c.startswith("C_Beers") for c in codes), codes


def test_good_case_no_harm():
    case = {
        "complaints": "боль в горле",
        "anamnesis_doctor": "болен 2 дня",
        "objective_status": "гиперемия зева",
        "clinical_diagnosis": "J02.9 Острый фарингит",
        "mkb_code_main": "J02.9",
        "exam_recommendations": "осмотр",
        "treatment_recommendations": "парацетамол 500 мг при температуре",
        "dispensary_info": "явка при ухудшении",
    }
    r = evaluate_kz_deep(case, drug_ctx=_drug_ctx())
    assert r["has_potential_harm"] is False
    assert r["overall_status"] in ("good", "acceptable", "review")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
