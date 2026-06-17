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


def test_semicolon_clinical_continuation_merged_with_icd():
    diags = parse_diagnoses(
        "M54.1. Радикулопатия; Вертеброгенная правосторонняя люмбоишиалгия, "
        "рефлекторный миотонический компонент, умеренный болевой синдром, острый период."
    )
    assert len(diags) == 1
    assert diags[0].icd10_code == "M54.1"


def test_junk_mkb_line_skipped():
    diags = parse_diagnoses("N72. Цервицит; Соп.: МКБ. НЖО 1 ст.; E03.9 Гипотиреоз")
    codes = [d.icd10_code for d in diags if d.icd10_code]
    assert "N72" in codes or any(c and c.startswith("N72") for c in codes)
    assert "E03.9" in codes
    assert len(diags) == 2
