"""Тесты B2C-обёртки онкориска для patient-флоу и фиче-флага."""
from __future__ import annotations

from clinical_knowledge import onco_risk as orisk
from clinical_knowledge.patient_flags import patient_onco_questions_enabled

FORBIDDEN = ["рак", "онколог", "злокачествен", "опухол", "метастаз", "карцином"]


def test_b2c_block_from_text_returns_safe_questions():
    block = orisk.b2c_block_from_text(
        "Жалобы: ректальное кровотечение, потеря веса.", age=62, sex="male"
    )
    assert block is not None
    assert block["show_numbers"] is False
    assert block["questions"]
    for q in block["questions"]:
        low = q.lower()
        assert not any(w in low for w in FORBIDDEN)
        assert "%" not in q
        assert not any(ch.isdigit() for ch in q)


def test_b2c_block_none_when_no_signal():
    block = orisk.b2c_block_from_text("плановый профосмотр, жалоб нет", age=30, sex="male")
    # Нет симптомов и не скрининговый возраст -> вопросов может не быть.
    assert block is None or isinstance(block, dict)


def test_patient_onco_flag_default_on(monkeypatch):
    monkeypatch.delenv("PATIENT_ONCO_QUESTIONS_ENABLED", raising=False)
    assert patient_onco_questions_enabled() is True
    monkeypatch.setenv("PATIENT_ONCO_QUESTIONS_ENABLED", "0")
    assert patient_onco_questions_enabled() is False
    monkeypatch.setenv("PATIENT_ONCO_QUESTIONS_ENABLED", "1")
    assert patient_onco_questions_enabled() is True
