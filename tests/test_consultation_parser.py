"""Тесты структурного парсера КЗ (ТЗ раздел 24)."""
from __future__ import annotations

import datetime as dt

from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.template_parser import parse_template_blocks

SAMPLE = """\
ООО Медицинский центр Здоровье
Врач: гастроэнтеролог Иванов И. И.
Дата консультации: 14.07.2024 10:30
Дата рождения: 12.05.1976
Пол: мужской

Жалобы: боли в эпигастрии, изжога после еды.
Анамнез: считает себя больным около 2 лет.
Объективный статус: состояние удовлетворительное.
Данные обследований: ФГДС от 01.07.2024 - эрозивный гастрит.
Диагноз: K29.7 Хронический гастрит, обострение.
Рекомендации по обследованию: УЗИ ОБП, общий анализ крови.
Рекомендации по лечению: Омепразол 20 мг 2 раза в день 14 дней.
Дата повторной явки: 28.07.2024
"""


def test_basic_extraction():
    doc = parse_consultation(SAMPLE, consultation_id="t1")
    assert doc.doctor_specialty and "гастроэнтеролог" in doc.doctor_specialty.lower()
    assert doc.consultation_date == dt.date(2024, 7, 14)
    assert doc.patient.birth_date == dt.date(1976, 5, 12)
    assert doc.patient.age_years == 48  # на 14.07.2024
    assert doc.patient.sex == "male"
    assert doc.patient.adult_or_child == "adult"


def test_sections_present():
    doc = parse_consultation(SAMPLE, consultation_id="t2")
    assert doc.sections.complaints and "эпигастри" in doc.sections.complaints.lower()
    assert doc.sections.anamnesis
    assert doc.sections.objective_status
    assert doc.sections.diagnosis_text


def test_diagnoses_exams_meds_followup():
    doc = parse_consultation(SAMPLE, consultation_id="t3")
    assert any(d.icd10_code == "K29.7" for d in doc.diagnoses)
    assert len(doc.planned_exams) >= 2
    assert len(doc.performed_exams) >= 1
    assert any(m.drug_name and "омепразол" in m.drug_name.lower() for m in doc.medications)
    assert doc.follow_up and doc.follow_up[0].date == dt.date(2024, 7, 28)


def test_quality_flags():
    doc = parse_consultation("Диагноз: Экзема ?\nundefined", consultation_id="t4")
    assert doc.extraction_quality.has_undefined
    assert doc.extraction_quality.has_question_mark_diagnosis


def test_template_blocks():
    text = (
        ">>> L30 Экзема кожи ?:\n"
        "* ОБСЛЕДОВАНИЯ ОБЯЗАТЕЛЬНЫЕ:\n"
        "- общий анализ крови\n"
        "- глюкоза крови\n"
        ">>> L93.0 Дискоидная красная волчанка:\n"
        "* ОБСЛЕДОВАНИЯ ДОПОЛНИТЕЛЬНЫЕ:\n"
        "- ANA\n"
    )
    blocks = parse_template_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].icd10_code == "L30"
    assert blocks[0].block_type == "required_exams"
    assert "общий анализ крови" in blocks[0].items
    assert blocks[1].icd10_code == "L93.0"
    assert blocks[1].block_type == "additional_exams"
