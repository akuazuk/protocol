"""Calm respectful questions with deny-list."""
from __future__ import annotations

from clinical_knowledge.patient_questions import (
    FORBIDDEN_PATTERNS_RU,
    build_calm_questions,
    is_forbidden_question,
    sanitize_question_text,
)


def test_forbidden_playful_phrases() -> None:
    assert is_forbidden_question("Анамнез как черновик на коленке - что дописать?")
    assert is_forbidden_question("Осмотр был, а половина пропала - что важно?")
    assert not is_forbidden_question("Подскажите, пожалуйста, в какие сроки выполнить МРТ?")


def test_calm_questions_for_neurology() -> None:
    kz = "МРТ шейного отдела. после: Пентоксифиллин. головная боль"
    structured = [
        {
            "id": "q1",
            "source_gap": "Нет срока МРТ",
            "block_id": "exams",
            "category_ru": "Обследования",
            "severity": "medium",
        }
    ]
    out = build_calm_questions(structured, kz_text=kz)
    texts = " ".join(q["text"] for q in out).lower()
    assert "мрт" in texts
    assert "после" in texts or "препарат" in texts
    for q in out:
        assert not is_forbidden_question(q["text"])
    assert len(FORBIDDEN_PATTERNS_RU) >= 5


def test_sanitize_strips_protocol_phrasing() -> None:
    assert "положено" not in sanitize_question_text("По протоколу положено УЗИ").lower()
