"""Лабораторные маркеры и сверка с КЗ (B2C)."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.lab_result_parser import detect_lab_panels, extract_lab_markers, format_marker_line
from clinical_knowledge.patient_lab_crosscheck import crosscheck_labs_with_kz
from clinical_knowledge.text_extract import extract_pdf_text_bytes

_KRAVIRA_SNIPPET = (
    "Биохимический анализ крови (БАК) "
    "Общий белок (TPROT), г/л 78,3 66 - 87 г/л "
    "Мочевина (UREA), ммоль/л 5,5 2.2 - 8.3 ммоль/л "
    "Глюкоза (GLUC), ммоль/л 5,57 3.9 - 6.2 ммоль/л "
    "Аспартатаминотransferase (AST), Ед/л 18,5 0 - 37 Ед/л"
)

_INVITRO_SNIPPET = (
    "ИНВИТРО АТ к нативной (двуспир.) ДНК IgG 2.70 МЕ/мл "
    "< 20 МЕ/мл - отрицательно"
)


def test_extract_lab_markers_oak() -> None:
    text = "ОАК: гемоглобин 132 г/л, лейкоциты 6.2, СОЭ 12 мм/ч"
    markers = extract_lab_markers(text)
    names = {m["marker"].lower() for m in markers}
    assert "гемоглобин" in names or "лейкоциты" in names


def test_extract_kravira_bak_rows() -> None:
    markers = extract_lab_markers(_KRAVIRA_SNIPPET)
    names = {m["marker"].lower() for m in markers}
    assert "мочевина" in names
    assert "глюкоза" in names
    assert any(m.get("value") for m in markers)
    assert "Приложение" not in names
    assert "200" not in names


def test_extract_invitro_row() -> None:
    markers = extract_lab_markers(_INVITRO_SNIPPET)
    assert markers
    line = format_marker_line(markers[0]).lower()
    assert "днк" in line or "igg" in line
    assert "2.7" in line


def test_detect_lab_panels() -> None:
    panels = detect_lab_panels(_KRAVIRA_SNIPPET)
    assert any("БАК" in p or "Биохим" in p for p in panels)


def test_crosscheck_missing_in_kz() -> None:
    kz = "Диагноз: ОРВИ. Рекомендовано наблюдение."
    lab = "СРБ (CRP), мг/л 24 0 - 6 мг/л, гемоглобин (HGB), г/л 140 120 - 160 г/л"
    out = crosscheck_labs_with_kz(kz_text=kz, lab_text=lab)
    assert out["lab_count"] >= 1
    assert out["missing_in_kz_lines"]
    assert out["markers_table"]
    assert out["summary_ru"]


def test_real_pdf_a1_no_form_junk() -> None:
    pdf = Path("clients_consult/a_1.pdf")
    if not pdf.is_file():
        return
    text, _, err = extract_pdf_text_bytes(pdf.read_bytes(), max_pages=20)
    assert not err
    markers = extract_lab_markers(text or "")
    names = [m["marker"] for m in markers]
    assert "мочевина" in names or "креатинин" in names
    assert not any(n in ("725", "Приложение", "200") for n in names)
    assert len(markers) >= 8


def test_real_pdf_invitro_a2() -> None:
    pdf = Path("clients_consult/A_2.pdf")
    if not pdf.is_file():
        return
    text, _, _ = extract_pdf_text_bytes(pdf.read_bytes(), max_pages=20)
    markers = extract_lab_markers(text or "")
    joined = " ".join(format_marker_line(m).lower() for m in markers)
    assert "днк" in joined or "igg" in joined
