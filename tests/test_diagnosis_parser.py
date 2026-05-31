"""Тесты разбора диагнозов (ТЗ раздел 24)."""
from __future__ import annotations

from clinical_knowledge.diagnosis_parser import has_malignancy_flag, parse_diagnoses


def _by_text(diags, needle):
    return next(d for d in diags if needle.lower() in d.raw_text.lower())


def test_icd_leading_codes():
    diags = parse_diagnoses(
        "K30. Диспепсия\nL93.0. Дискоидная красная волчанка\n"
        "I80.1. Флебит и тромбофлебит бедренной вены"
    )
    assert _by_text(diags, "Диспепсия").icd10_code == "K30"
    assert _by_text(diags, "волчанка").icd10_code == "L93.0"
    assert _by_text(diags, "Флебит").icd10_code == "I80.1"


def test_suspected_marker():
    diags = parse_diagnoses("Дискоидная красная волчанка ?")
    assert diags[0].certainty == "suspected"


def test_malignancy_flag():
    assert has_malignancy_flag("Нельзя исключить инвазию")
    assert not has_malignancy_flag("Гастрит, ремиссия")


def test_multiple_diagnoses_separate():
    diags = parse_diagnoses("Основной: K30 Диспепсия; Сопутствующий: I10 Гипертензия")
    assert len(diags) == 2
    assert diags[0].diagnosis_role == "primary"
