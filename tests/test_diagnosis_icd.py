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
