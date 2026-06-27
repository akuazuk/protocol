"""Tests for patient question builder."""
from __future__ import annotations

from clinical_knowledge.patient_question_builder import build_useful_patient_questions


def test_derm_questions_no_generic_anamnesis() -> None:
    kz = "Жалобы: высыпания. Диагноз L93.0. ОАК, ОАМ. Гидроксихлорохин 2 недели."
    clarify = [
        {"topic_ru": "Анализы", "text_ru": "когда сдать общий анализ крови"},
        {"topic_ru": "Лечение", "text_ru": "сколько дней принимать препараты"},
    ]
    exams = [
        {"exam_type": "LAB_OAK", "category": "lab", "label_ru": "Общий анализ крови (ОАК)"},
        {"exam_type": "LAB_OAM", "category": "lab", "label_ru": "Общий анализ мочи (ОАМ)"},
    ]
    gaps = [
        {
            "source_gap": "Аллергоанамнез не отражён",
            "block_id": "anamnesis",
            "category_ru": "Анамнез",
        }
    ]
    out = build_useful_patient_questions(
        kz_text=kz,
        clarification_points=clarify,
        exams=exams,
        structured_gaps=gaps,
    )
    blob = " ".join(q["text"].lower() for q in out)
    assert "аллергоанамнез" not in blob
    assert "локализац" not in blob
    assert len(out) <= 5
    assert any("анализ" in q["text"].lower() or "оак" in q["text"].lower() for q in out)


def test_lab_crosscheck_question() -> None:
    lc = {
        "panels_ru": ["Общий анализ мочи (ОАМ)"],
        "missing_in_kz_lines": ["Лейкоциты 1 ↑", "Белок отрицательный"],
    }
    out = build_useful_patient_questions(kz_text="Диагноз L93", lab_crosscheck=lc)
    assert any("анализ" in q["text"].lower() or "лечени" in q["text"].lower() for q in out)
