"""Тесты B2B-advisory онконастороженности в consult-review (за флагом)."""
from __future__ import annotations

import rag_server as rs


def test_demographics_from_text_age_words_and_sex():
    age, sex, aoc = rs._onco_demographics_from_text(
        "Пациент мужского пола, 62 года. Жалобы на боль."
    )
    assert age == 62
    assert sex == "male"
    assert aoc == "adult"


def test_demographics_child_when_under_18():
    age, sex, aoc = rs._onco_demographics_from_text("Ребёнок, 8 лет, женский пол.")
    assert age == 8
    assert sex == "female"
    assert aoc == "child"


def test_demographics_unknown_when_absent():
    age, sex, aoc = rs._onco_demographics_from_text("Жалобы на кашель.")
    assert age is None
    assert sex == "unknown"
    assert aoc == "adult"


def test_attach_respects_flag(monkeypatch):
    monkeypatch.setenv("CONSULT_ONCO_RISK_ADVISORY_ENABLED", "0")
    res: dict = {}
    rs._consult_attach_onco_risk(res, "мужчина 70 лет, кровохарканье")
    assert "onco_risk" not in res

    monkeypatch.setenv("CONSULT_ONCO_RISK_ADVISORY_ENABLED", "1")
    res2: dict = {}
    rs._consult_attach_onco_risk(
        res2,
        "Мужчина 62 года. Ректальное кровотечение, потеря веса, боль в животе.",
    )
    assert "onco_risk" in res2
    o = res2["onco_risk"]
    assert o["context"] == "symptomatic"
    assert o["triage_level"] == "refer"
    assert any(s["site"] == "colorectal" for s in o["sites"])
    assert "не диагноз" in o["method_note_ru"]


def test_attach_noop_on_empty_text(monkeypatch):
    monkeypatch.setenv("CONSULT_ONCO_RISK_ADVISORY_ENABLED", "1")
    res: dict = {}
    rs._consult_attach_onco_risk(res, "")
    assert "onco_risk" not in res
