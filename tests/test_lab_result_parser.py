"""Лабораторные маркеры и сверка с КЗ (B2C)."""
from __future__ import annotations

from clinical_knowledge.lab_result_parser import extract_lab_markers
from clinical_knowledge.patient_lab_crosscheck import crosscheck_labs_with_kz


def test_extract_lab_markers_oak() -> None:
    text = "ОАК: гемоглобин 132 г/л, лейкоциты 6.2, СОЭ 12 мм/ч"
    markers = extract_lab_markers(text)
    names = {m["marker"].lower() for m in markers}
    assert "оак" in names or "гемоглобин" in names
    assert "лейкоциты" in names


def test_crosscheck_missing_in_kz() -> None:
    kz = "Диагноз: ОРВИ. Рекомендовано наблюдение."
    lab = "СРБ 24 мг/л, гемоглобин 140 г/л"
    out = crosscheck_labs_with_kz(kz_text=kz, lab_text=lab)
    assert out["lab_count"] >= 1
    assert out["missing_in_kz"]
    assert out["notes_ru"]
