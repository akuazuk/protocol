"""Тесты словаря нозология→МКБ и гейтинга симптом-кодов (ТЗ: точный подбор КП)."""
from __future__ import annotations

from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.diagnosis_icd import (
    enrich_diagnosis_codes,
    has_disease_code,
    is_symptom_code,
    lookup_disease_icd,
    prioritize_codes,
)


def test_lookup_discoid_lupus():
    assert lookup_disease_icd("Дискоидная красная волчанка ?")[0] == "L93.0"


def test_lookup_specificity_order():
    # Системная форма не должна перебиваться общей «красная волчанка».
    assert lookup_disease_icd("Системная красная волчанка")[0] == "M32.9"


def test_symptom_code_detection():
    assert is_symptom_code("R21.9") is True
    assert is_symptom_code("Z00.0") is True
    assert is_symptom_code("L93.0") is False


def test_prioritize_disease_over_symptom():
    assert prioritize_codes(["R21.9", "L93.0"]) == ["L93.0", "R21.9"]


def test_has_disease_code():
    assert has_disease_code(["R21.9"]) is False
    assert has_disease_code(["R21.9", "L93.0"]) is True


def test_enrich_adds_disease_code():
    codes, meta = enrich_diagnosis_codes("Дискоидная красная волчанка ?", ["R21.9"])
    assert codes[0] == "L93.0"
    assert "L93.0" in meta["added_from_lexicon"]
    assert meta["had_only_symptom_codes"] is True


def test_parser_assigns_disease_code_when_missing():
    text = (
        "Дерматолог\nДата: 14.07.2024\n"
        "Ф.И.О: Иванов Павел Леонидович, 12.07.1976.\n"
        "Диагноз: Дискоидная красная волчанка ?\n"
        "Рекомендации по обследованию: общий анализ крови."
    )
    doc = parse_consultation(text)
    assert doc.diagnoses
    assert doc.diagnoses[0].icd10_code == "L93.0"


def test_vich_not_matched_inside_patronymic_or_pervichnyj():
    assert lookup_disease_icd("первичный ЛОР-осмотр") == []
    assert lookup_disease_icd("Некраш Станислав Юрьевич") == []
    assert lookup_disease_icd("ВИЧ-инфекция")[0] == "B24"
    assert lookup_disease_icd("подтверждён ВИЧ")[0] == "B24"


def test_sticky_medical_exam_parses_h60_not_b24():
    """Медосмотр = расширенное КЗ: слипшиеся заголовки PDF не должны ломать МКБ."""
    text = (
        "МЕДИЦИНСКИЙ ОСМОТР ЛОР-ВРАЧпервичныйЛОР-врач\n"
        "Дата и время проведения медицинского осмотра: 01.06.2026 10:30"
        "Ф.И.О: Некраш Станислав Юрьевич, 02.12.1992."
        "Жалобы пациента: На боль в левом ухе"
        "Анамнез заболевания: Болеет около 4х суток"
        "Данные результатов медицинского осмотра:Осмотрен на чесотку, ped at scab abs."
        "Вес 123 кг. Температура тела 37.2 °С."
        "AD=AS - m/t - п/серая.Голосовая щель широкая."
        "Диагноз: H60. Наружный отит;Острый левосторонний наружный отит."
        "Рекомендации: Неладекс по 4 капли * 3р/день в правое ухо 7 дней"
        "Дата повторной явки: 05.06.2026"
        "Врач: Яровой Иван Юрьевич, врач высшей категории"
    )
    doc = parse_consultation(text, consultation_id="med_exam")
    assert doc.sections.diagnosis_text and "H60" in doc.sections.diagnosis_text
    assert doc.sections.complaints and "ухе" in doc.sections.complaints.lower()
    assert doc.sections.anamnesis
    assert doc.consultation_date == __import__("datetime").date(2026, 6, 1)
    assert doc.diagnoses
    assert doc.diagnoses[0].icd10_code == "H60"
    assert not any(d.icd10_code == "B24" for d in doc.diagnoses)


def test_short_kz_same_h60():
    text = (
        "КОНСУЛЬТАТИВНОЕ ЗАКЛЮЧЕНИЕ ЛОР-врач\n"
        "Дата: 01.06.2026 10:30\n"
        "Ф.И.О: Некраш Станислав Юрьевич, 02.12.1992.\n"
        "Жалобы: На боль в левом ухе\n"
        "Анамнез: Болеет около 4х суток\n"
        "Диагноз: H60. Наружный отит; Острый левосторонний наружный отит.\n"
        "Рекомендации: Неладекс по 4 капли\n"
        "Дата повторной явки: 05.06.2026\n"
    )
    doc = parse_consultation(text, consultation_id="kz_short")
    assert doc.diagnoses[0].icd10_code == "H60"
