"""Дата консультации: ISO из MIS и ДД.ММ.ГГГГ из PDF/visit_date_text."""
from __future__ import annotations

import datetime as dt

from clinical_knowledge.consult_parser import parse_consultation
from clinical_knowledge.date_parser import format_date_dmy, parse_date
from scripts.run_mis_protocol_l1_batch import build_kz_text


def test_parse_date_iso_and_dmy():
    assert parse_date("2026-07-13") == dt.date(2026, 7, 13)
    assert parse_date("13.07.2026") == dt.date(2026, 7, 13)
    assert format_date_dmy(dt.date(2026, 7, 13)) == "13.07.2026"


def test_build_kz_text_prefers_visit_date_text():
    row = {
        "date": "2026-07-13",
        "visit_date_text": "13.07.2026",
        "doctor_fio": "Анисимов Игорь Анатольевич",
        "doctor_specialization": "Невролог",
        "complaints": "боль в шее",
    }
    text = build_kz_text(row)
    assert text.startswith("Дата консультации: 13.07.2026")
    doc = parse_consultation(text)
    assert doc.consultation_date == dt.date(2026, 7, 13)
    assert not doc.extraction_quality.has_missing_consultation_date


def test_build_kz_text_normalizes_iso_when_no_visit_date_text():
    row = {
        "date": "2026-07-13",
        "visit_date_text": "",
        "complaints": "боль",
    }
    text = build_kz_text(row)
    assert "13.07.2026" in text.splitlines()[0]
    doc = parse_consultation(text)
    assert doc.consultation_date == dt.date(2026, 7, 13)


def test_pdf_style_exam_datetime_line():
    text = (
        "Дата и время проведения медицинского осмотра: 13.07.2026 14:30\n"
        "Ф.И.О: Задора Екатерина Леонидовна, 16.09.1971.\n"
        "Жалобы пациента: боль в шее\n"
    )
    doc = parse_consultation(text)
    assert doc.consultation_date == dt.date(2026, 7, 13)
