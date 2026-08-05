"""Эталон concordance: обезличенный fact-graph МО Смирнова + негативный контроль."""
from __future__ import annotations

import os

from clinical_knowledge.kz_deep_eval import evaluate_kz_deep
from clinical_knowledge.mo_case_signals import extract_mo_case_signals
from clinical_knowledge.mo_concordance_findings import evaluate_mo_concordance

# Обезличенный разбор Downloads/KZ/smirnova.pdf (без ФИО/идентификаторов).
SMIRNOVA_CASE = {
    "patient_age_years": 9,
    "complaints": "на хромоту на правую ногу. Болеет 3 месяца.",
    "anamnesis_doctor": "Со слов травмы не было. Аллергоанамнез не отягощён.",
    "objective_status": (
        "Коленные суставы левый без особенностей, правый отёчен, движения в полном объёме "
        "разгибание/сгибание: 0/0/140 (140°), безболезненны. "
        "St. localis: боли при пальпации прямой мышцы правого бедра."
    ),
    "clinical_diagnosis": "M60. Миозит; Миозит прямой мышцы правого бедра.",
    "mkb_code_main": "M60",
    "treatment_recommendations": (
        "Ибуфен Д 10 мл. 3 раза в день 10 дней. "
        "Массаж мышц обеих бедер №10. "
        "Осмотр травматолога - ортопеда при отрицательной динамики."
    ),
    "exam_recommendations": "",
}

CLEAN_ADULT_CASE = {
    "patient_age_years": 42,
    "complaints": "боль в горле 2 дня",
    "anamnesis_doctor": (
        "Заболел остро после переохлаждения. Температуры не было. "
        "Динамика без ухудшения. Нагрузки спортивные отрицает. Травм не было."
    ),
    "objective_status": "Зев гиперемирован. Суставы без особенностей, отёков нет.",
    "clinical_diagnosis": "J02.9 Острый фарингит неуточнённый",
    "mkb_code_main": "J02.9",
    "treatment_recommendations": "Полоскание, местный антисептик 5 дней. Явка к терапевту через 5 дней.",
    "exam_recommendations": "ОАК при сохранении температуры",
}


def test_smirnova_signals_detect_knee_edema_and_duration() -> None:
    sig = extract_mo_case_signals(SMIRNOVA_CASE)
    assert sig["audience"] == "pediatric"
    assert sig["has_limp"] is True
    assert sig["duration_days"] == 90
    assert any(item["joint"] == "knee" for item in sig["joint_edema"])
    assert sig["plan_bilateral"] is True
    assert sig["follow_up_on_worsening_only"] is True
    assert sig["plan_has_imaging"] is False


def test_smirnova_concordance_expected_codes() -> None:
    codes = {f["code"] for f in evaluate_mo_concordance(SMIRNOVA_CASE)}
    expected = {
        "finding_not_in_diagnosis",
        "anamnesis_thin_for_duration",
        "underworkup_chronic_red_flag",
        "plan_laterality_mismatch",
        "icd_weakly_supported",
        "pediatric_limp_ddx_not_addressed",
    }
    missing = expected - codes
    assert not missing, f"missing={missing}, got={codes}"
    assert len(expected & codes) >= 4


def test_clean_adult_case_has_no_smirnova_p1_patterns() -> None:
    codes = {f["code"] for f in evaluate_mo_concordance(CLEAN_ADULT_CASE)}
    assert "finding_not_in_diagnosis" not in codes
    assert "underworkup_chronic_red_flag" not in codes
    assert "pediatric_limp_ddx_not_addressed" not in codes


def test_adult_chronic_non_msk_not_thin_anamnesis() -> None:
    """E2: anamnesis_thin не должен срабатывать на хронике без хромоты/отёка."""
    case = {
        "patient_age_years": 40,
        "complaints": "головная боль 3 месяца",
        "anamnesis_doctor": "Аллергоанамнез не отягощён. Наследственность не отягощена.",
        "objective_status": "Неврологически без очага. Суставы без особенностей.",
        "clinical_diagnosis": "G43.9 Мигрень неуточнённая",
        "mkb_code_main": "G43.9",
        "treatment_recommendations": "НПВП по потребности. Явка через месяц.",
        "exam_recommendations": "",
    }
    codes = {f["code"] for f in evaluate_mo_concordance(case)}
    assert "anamnesis_thin_for_duration" not in codes


def test_edema_negation_oteki_net_not_joint_edema() -> None:
    case = {
        "patient_age_years": 44,
        "complaints": "Боль в правом бедре 1 месяц",
        "anamnesis_doctor": "Травму отрицает. Аллергоанамнез не отягощён.",
        "objective_status": (
            "Отеки: нет. Движения в правом тазобедренном суставе в полном объеме. "
            "Осевая нагрузка безболезненна."
        ),
        "clinical_diagnosis": "M16 Коксартроз",
        "mkb_code_main": "M16",
        "treatment_recommendations": "НПВП 7 дней. Контроль через 2 недели.",
        "exam_recommendations": "Рентген ТБС",
    }
    sig = extract_mo_case_signals(case)
    assert sig["joint_edema"] == []
    codes = {f["code"] for f in evaluate_mo_concordance(case)}
    assert "finding_not_in_diagnosis" not in codes
    assert "anamnesis_thin_for_duration" not in codes


def test_shadow_findings_do_not_change_overall_by_default(monkeypatch) -> None:
    monkeypatch.setenv("MO_CONCORDANCE_FINDINGS", "1")
    monkeypatch.setenv("MO_CONCORDANCE_PRIMARY", "0")
    deep = evaluate_kz_deep(SMIRNOVA_CASE, protocol_ctx=None, drug_ctx={})
    shadow_codes = {f["code"] for f in (deep.get("shadow_findings") or [])}
    primary_codes = {f["code"] for f in (deep.get("findings") or [])}
    assert "finding_not_in_diagnosis" in shadow_codes
    assert "finding_not_in_diagnosis" not in primary_codes
    # overall считается без shadow
    assert deep.get("overall_pct") is not None


def test_flag_off_skips_shadow(monkeypatch) -> None:
    monkeypatch.setenv("MO_CONCORDANCE_FINDINGS", "0")
    deep = evaluate_kz_deep(SMIRNOVA_CASE, protocol_ctx=None, drug_ctx={})
    assert deep.get("shadow_findings") == []
