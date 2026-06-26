"""Тон вопросов врачу (B2C) - три различимых стиля."""
from __future__ import annotations

from clinical_knowledge.patient_question_tone import (
    QUESTION_TONE_CATALOG,
    apply_tone_to_questions,
    normalize_question_tone,
    question_tones_for_api,
    render_doctor_question,
)
from clinical_knowledge.patient_report import build_patient_report


def test_normalize_tone_aliases() -> None:
    assert normalize_question_tone("официально") == "official"
    assert normalize_question_tone("шуточно") == "playful"
    assert normalize_question_tone("light") == "playful"
    assert normalize_question_tone("friendly") == "serious"
    assert normalize_question_tone("") == "serious"


def test_catalog_has_three_distinct_tones() -> None:
    ids = [row["id"] for row in QUESTION_TONE_CATALOG]
    assert ids == ["serious", "official", "playful"]
    assert len(question_tones_for_api()) == 3
    assert any(row.get("default") for row in QUESTION_TONE_CATALOG)


def test_same_intent_differs_by_tone() -> None:
    gap = "Нет длительности терапии"
    serious, _ = render_doctor_question(gap=gap, block_id="treatment", tone="serious")
    official, _ = render_doctor_question(gap=gap, block_id="treatment", tone="official")
    playful, _ = render_doctor_question(gap=gap, block_id="treatment", tone="playful")
    assert serious != official != playful
    assert all(q.endswith("?") for q in (serious, official, playful))
    assert "Вы" in official or "прошу" in official.lower()
    assert "пенсии" in playful.lower() or "числа" in playful.lower()


def test_playful_tone_respectful() -> None:
    text, _ = render_doctor_question(
        gap="нет УЗИ",
        block_id="exams",
        category_ru="Обследования",
        tone="playful",
    )
    assert "?" in text
    assert "дурак" not in text.lower()
    assert "идиот" not in text.lower()


def test_apply_tone_to_structured_questions() -> None:
    raw = [
        {
            "id": "q1",
            "source_gap": "Нет длительности терапии",
            "block_id": "treatment",
            "category_ru": "Лечение",
            "severity": "high",
        }
    ]
    out = apply_tone_to_questions(raw, "official")
    assert out[0]["text"]
    assert out[0]["tone"] == "official"
    assert out[0].get("intent") == "treatment_duration"


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
    assert "Прошу" in rep["action_checklist"][0]["text"]
