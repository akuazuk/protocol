"""Тесты движка онкориска: байес, пороги, полнота, безопасность B2C."""
from __future__ import annotations

import math

import pytest

from clinical_knowledge import onco_risk as orisk
from clinical_knowledge.onco_risk import OncoInputs, assess


def test_extract_features_strata_filters_sex_and_age():
    # PSA - только мужчины 50+. У женщины не должно матчиться.
    male = OncoInputs(text="осмотр", labs_text="PSA 8", age=65, sex="male")
    female = OncoInputs(text="осмотр", labs_text="PSA 8", age=65, sex="female")
    ids_m = {f.id for f in orisk.extract_features(male)}
    ids_f = {f.id for f in orisk.extract_features(female)}
    assert "psa_high" in ids_m
    assert "psa_high" not in ids_f


def test_bayes_single_feature_increases_over_baseline():
    inp = OncoInputs(text="кровохарканье", age=70, sex="male", symptom_duration_known=True)
    a = assess(inp)
    lung = next(s for s in a.sites if s.site == "lung")
    # Базовый риск лёгкого 0.0018; кровохарканье LR=13 должно поднять заметно.
    assert lung.p > 0.0018
    assert lung.ci_low <= lung.p <= lung.ci_high


def test_ppv_to_lr_monotonic():
    odds0 = 0.001 / (1 - 0.001)
    lr_low = orisk._ppv_to_lr(0.02, odds0)
    lr_high = orisk._ppv_to_lr(0.10, odds0)
    assert lr_high > lr_low > 1.0


def test_secondary_features_are_shrunk():
    # Один сильный признак vs он же + слабые: рост есть, но не мультипликативно полный.
    one = assess(OncoInputs(text="ректальное кровотечение", age=60, sex="male"))
    many = assess(OncoInputs(text="ректальное кровотечение, боль в животе, диарея",
                             age=60, sex="male"))
    p_one = next(s.p for s in one.sites if s.site == "colorectal")
    p_many = next(s.p for s in many.sites if s.site == "colorectal")
    assert p_many > p_one


def test_triage_threshold_3pct():
    assert orisk.triage_level(0.05) == "refer"
    assert orisk.triage_level(0.02) == "low_not_no"
    assert orisk.triage_level(0.005) == "low"
    # У детей порог ниже.
    assert orisk.triage_level(0.015, is_child=True) == "refer"


def test_completeness_bands():
    rich = OncoInputs(text="кровохарканье", age=70, sex="male",
                      symptom_duration_known=True, labs_text="тромбоцитоз",
                      smoking=True, family_history=True, bmi=24.0)
    poor = OncoInputs(text="кашель")
    c_rich = orisk.data_completeness(rich, orisk.extract_features(rich))
    c_poor = orisk.data_completeness(poor, orisk.extract_features(poor))
    assert c_rich.score > c_poor.score
    assert c_rich.band == "quantitative"
    assert c_poor.band == "qualitative_only"


def test_low_completeness_skips_quantitative():
    a = assess(OncoInputs(text="кашель"))
    assert a.completeness.band == "qualitative_only"
    assert a.sites == []
    assert a.b2c_questions  # вопросы всё равно есть


def test_surveillance_context_no_quantitative_no_symptom_questions():
    a = assess(OncoInputs(text="состояние после химиотерапии, на учёте у онколога",
                          age=55, sex="female"))
    assert a.context == "surveillance"
    assert a.sites == []
    assert all("обследование кишечника" not in q for q in a.b2c_questions)


def test_b2c_no_forbidden_words():
    forbidden = ["рак", "онколог", "злокачествен", "опухол", "метастаз", "карцином"]
    cases = [
        OncoInputs(text="ректальное кровотечение, потеря веса", age=62, sex="male"),
        OncoInputs(text="кровохарканье", age=70, sex="male"),
        OncoInputs(text="желтуха", age=66, sex="female"),
        OncoInputs(text="состояние после химиотерапии", age=50, sex="male"),
        OncoInputs(text="плановый осмотр", age=55, sex="female"),
    ]
    for inp in cases:
        a = assess(inp)
        for q in a.b2c_questions:
            low = q.lower()
            assert not any(w in low for w in forbidden), f"scary word in: {q}"


def test_b2c_no_numbers_or_percent():
    a = assess(OncoInputs(text="ректальное кровотечение, потеря веса", age=62, sex="male"))
    for q in a.b2c_questions:
        assert "%" not in q
        assert not any(ch.isdigit() for ch in q), f"digit in B2C: {q}"


def test_any_cancer_risk_combination():
    sites = [orisk.SiteRisk("a", 0.2, 0.1, 0.3), orisk.SiteRisk("b", 0.1, 0.05, 0.2)]
    p = orisk.any_cancer_risk(sites)
    assert math.isclose(p, 1 - 0.8 * 0.9, rel_tol=1e-6)


def test_p_cap_not_exceeded():
    inp = OncoInputs(text="ректальное кровотечение, потеря веса, боль в животе, диарея",
                     age=75, sex="female", labs_text="кал на скрытую кровь, анемия",
                     symptom_duration_known=True)
    a = assess(inp)
    for s in a.sites:
        assert s.p <= 0.9 + 1e-9


def test_context_pediatric():
    a = assess(OncoInputs(text="увеличение лимфоузлов", age=8, sex="male",
                          adult_or_child="child"))
    assert a.context == "pediatric"
    assert a.b2c_questions


def test_advisory_note_present():
    a = assess(OncoInputs(text="кровохарканье", age=70, sex="male"))
    assert "не диагноз" in a.advisory_note
