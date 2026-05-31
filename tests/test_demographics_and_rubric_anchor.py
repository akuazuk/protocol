"""Регрессы по реальному кейсу: дерматологическое КЗ (дискоидная красная волчанка).

Покрывает баги: возраст из «болеет около 1 года», пол по отчеству врача вместо
пациента, нераспознанные дата консультации/ДР, привязка рубрики к чужой
специальности.
"""
from __future__ import annotations

from clinical_knowledge.age_sex_resolver import detect_sex, detect_sex_from_name, resolve_age
from clinical_knowledge.consult_analysis import analyze_consultation_text
from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.rubric_extractors import specialty_to_rubric

SAMPLE = """Дерматолог
Дата: 14.07.2024 15:30
Ф.И.О: Иванов Павел Леонидович, 12.07.1976.
Жалобы: на высыпания а коже шеи и затылка
Aнамнез: Болеет около 1 года. Наружно применял крем.
Объективный статус: на коже шеи атрофические рубцы, эритематозные высыпания.
Диагноз: Дискоидная красная волчанка ?
Рекомендации по обследованию: общий анализ крови, СОЭ.
Рекомендации по лечению: Гидроксихлорохин 200 мг 2 раза/сутки.
Врач-дерматовенеролог: Саутина Виктория Васильевна
"""


def test_age_not_taken_from_disease_duration():
    # «болеет около 1 года» не должно стать возрастом.
    info = resolve_age("Болеет около 1 года.")
    assert info["age_years"] is None


def test_age_from_birth_and_consult_dates():
    doc = parse_consultation(SAMPLE)
    assert doc.patient.birth_date is not None
    assert doc.consultation_date is not None
    assert doc.patient.age_years == 48
    assert doc.patient.age_group == "adult"
    assert doc.patient.adult_or_child == "adult"


def test_sex_from_patient_name_not_doctor():
    # Пациент — Павел Леонидович (м), врач — Виктория Васильевна (ж).
    doc = parse_consultation(SAMPLE)
    assert doc.patient.sex == "male"


def test_detect_sex_from_name():
    assert detect_sex_from_name("Иванов Павел Леонидович") == "male"
    assert detect_sex_from_name("Петрова Анна Сергеевна") == "female"
    assert detect_sex_from_name("") == "unknown"


def test_detect_sex_ignores_generic_patient_word():
    # «пациент устно проинформирован» больше не делает всех мужчинами.
    assert detect_sex("пациент устно проинформирован о вмешательстве") == "unknown"


def test_full_name_extracted():
    doc = parse_consultation(SAMPLE)
    assert doc.patient.full_name and "Павел" in doc.patient.full_name


def test_specialty_to_rubric_map():
    assert specialty_to_rubric("Дерматолог") == "dermatovenerologiya"
    assert specialty_to_rubric("Врач-гастроэнтеролог") == "gastroenterologiya"
    assert specialty_to_rubric("Кардиолог") == "bolezni-sistemy-krovoobrashcheniya"
    assert specialty_to_rubric("терапевт") is None


def test_rubric_anchored_to_doctor_specialty():
    out = analyze_consultation_text(SAMPLE)
    rs = out["rubric_specifics"]
    # Рубрика должна быть дерматовенерология, а не случайная (гастро/анестезио).
    assert rs["rubrics"], "rubrics must not be empty"
    assert rs["rubrics"][0] == "dermatovenerologiya"
    # Никаких посторонних рубрик с 0% впереди профильной.
    assert "gastroenterologiya" not in rs["rubrics"]


def test_matches_scoped_to_dermatology():
    out = analyze_consultation_text(SAMPLE)
    paths = [m.get("source_path", "") for m in out["matches"]]
    assert paths, "expected at least one match"
    assert all("dermatovenerologiya" in p for p in paths)
    # Дедуп: один и тот же протокол не повторяется.
    assert len(paths) == len(set(paths))
