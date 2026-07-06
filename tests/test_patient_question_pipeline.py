"""Единый пайплайн вопросов врачу (B2C)."""
from __future__ import annotations

from clinical_knowledge.patient_question_pipeline import (
    attach_questions_to_report,
    build_patient_doctor_questions,
)
from clinical_knowledge.patient_question_tone import is_playful_meta_template

KZ = """
Консультативное заключение
Жалобы: кашель, насморк.
Диагноз: J06.9 ОРВИ.
Рекомендации: парацетамол при температуре, обильное питьё.
"""

KZ_ALLERGY = """
Консультативное заключение
Диагноз: аллергический ринит?
Обследования: КТ, УЗИ
Анализы: ОАК, ОАМ
Лечение: Форте Аллергия, Форте
"""

MEDS_FORTE = [
    {"name": "Форте Аллергия", "clarity_issues": ["duration_missing"]},
    {"name": "Форте", "clarity_issues": []},
]

EXAMS_MIX = [
    {"label_ru": "КТ", "category": "imaging", "exam_type": "CT"},
    {"label_ru": "УЗИ", "category": "imaging", "exam_type": "US"},
    {"label_ru": "ОАК", "category": "lab", "exam_type": "LAB_OAK"},
    {"label_ru": "ОАМ", "category": "lab", "exam_type": "LAB_OAM"},
]


def test_playful_differs_from_serious_on_same_kz() -> None:
    serious = build_patient_doctor_questions(kz_text=KZ, question_tone="serious", limit=5)
    playful = build_patient_doctor_questions(kz_text=KZ, question_tone="playful", limit=5)
    assert len(serious) >= 1
    assert len(playful) >= 1
    s_texts = {q["text"] for q in serious}
    p_texts = {q["text"] for q in playful}
    assert s_texts != p_texts or any(q.get("tone") == "playful" for q in playful)


def test_questions_have_plain_context_or_why() -> None:
    out = build_patient_doctor_questions(kz_text=KZ, question_tone="serious", limit=5)
    assert any(q.get("why_ru") for q in out)
    assert all("?" in q.get("text", "") for q in out)


def test_discuss_first_marks_high_priority() -> None:
    lc = {
        "panels_ru": ["ОАМ"],
        "missing_in_kz_lines": ["Лейкоциты ↑"],
    }
    out = build_patient_doctor_questions(
        kz_text="Диагноз L93. Жалобы: высыпания.",
        lab_crosscheck=lc,
        question_tone="serious",
        limit=5,
    )
    assert any(q.get("discuss_first") for q in out)


def test_attach_questions_sets_tone_meta() -> None:
    qs = build_patient_doctor_questions(kz_text=KZ, question_tone="official", limit=3)
    rep = attach_questions_to_report({}, qs, question_tone="official")
    assert rep["question_tone"] == "official"
    assert rep["questions_intro_ru"]
    assert len(rep["action_checklist"]) == len(qs)
    assert "Прошу" in rep["action_checklist"][0]["text"] or "?" in rep["action_checklist"][0]["text"]


def test_no_forbidden_phrases_in_playful() -> None:
    from clinical_knowledge.patient_questions import is_forbidden_question

    out = build_patient_doctor_questions(kz_text=KZ, question_tone="playful", limit=5)
    for q in out:
        assert not is_forbidden_question(q.get("text") or "")


def test_playful_no_meta_templates_on_rich_kz() -> None:
    out = build_patient_doctor_questions(
        kz_text=KZ_ALLERGY,
        exams=EXAMS_MIX,
        meds=MEDS_FORTE,
        question_tone="playful",
        limit=5,
    )
    texts = [q.get("text") or "" for q in out]
    whys = [q.get("why_ru") or "" for q in out]
    assert len(out) >= 2
    for t in texts:
        assert not is_playful_meta_template(t)
        assert "намёк" not in t.lower()
    assert not any("хочу понять: в заключении" in w.lower() for w in whys)
    treatment = [q for q in out if q.get("block_id") == "treatment"]
    assert len(treatment) <= 1
    assert any("кт" in t.lower() or "узи" in t.lower() for t in texts)
    assert any("оак" in t.lower() or "оам" in t.lower() or "пробир" in t.lower() for t in texts)


def test_playful_questions_differ_between_kz() -> None:
    a = build_patient_doctor_questions(kz_text=KZ, question_tone="playful", limit=4)
    b = build_patient_doctor_questions(
        kz_text=KZ_ALLERGY,
        exams=EXAMS_MIX,
        meds=MEDS_FORTE,
        question_tone="playful",
        limit=4,
    )
    a_texts = {q["text"] for q in a}
    b_texts = {q["text"] for q in b}
    assert a_texts != b_texts
