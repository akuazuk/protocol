"""Тон вопросов врачу (B2C)."""
from __future__ import annotations

from clinical_knowledge.patient_question_tone import (
    apply_question_tone,
    normalize_question_tone,
    questions_panel_intro_ru,
)
from clinical_knowledge.patient_report import build_patient_report


def test_normalize_tone_aliases() -> None:
    assert normalize_question_tone("официально") == "official"
    assert normalize_question_tone("шуточно") == "light"
    assert normalize_question_tone("") == "friendly"


def test_apply_official_tone() -> None:
    base = "Подскажите, пожалуйста, как принимать препараты?"
    out = apply_question_tone(base, "official", block_id="treatment")
    assert "Вы" in out or "вы" in out.lower()
    assert out.endswith("?")


def test_apply_light_tone_respectful() -> None:
    base = "Нужно ли пересдать общий анализ крови?"
    out = apply_question_tone(base, "light", block_id="exams", category_ru="Обследования")
    assert "?" in out
    assert "дурак" not in out.lower()


def test_build_report_includes_tone_meta() -> None:
    l1 = {
        "confidence_score": 70,
        "matched_protocols_count": 1,
        "alignment": {
            "alignment_mean_score": 60,
            "alignment_cards": [
                {
                    "block_id": "treatment",
                    "name_ru": "Лечение",
                    "score_pct": 45,
                    "comment_ru": "Доза не детализирована.",
                    "gaps_ru": ["Нет длительности терапии"],
                },
            ],
        },
    }
    rep = build_patient_report(l1, question_tone="official")
    assert rep["question_tone"] == "official"
    assert rep["question_tone_meta"]["label_ru"]
    assert rep["questions_intro_ru"]
    assert rep["action_checklist"]
    assert rep["action_checklist"][0].get("emoji")
