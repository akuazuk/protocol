"""Тесты разбора лекарственных назначений (ТЗ раздел 24)."""
from __future__ import annotations

import datetime as dt

from clinical_knowledge.medication_parser import RE_SCHEDULE_PREFIX, parse_medications, looks_like_medication_item


def test_medication_regex_compiles():
    assert RE_SCHEDULE_PREFIX.pattern


def test_rivaroxaban():
    meds = parse_medications("Ривороксабан 20 мг раз в день постоянно")
    m = meds[0]
    assert m.dose_value == 20.0
    assert m.dose_unit == "мг"
    assert m.duration and "постоянно" in m.duration.lower()
    assert m.frequency and "раз" in m.frequency.lower()


def test_hydroxychloroquine():
    meds = parse_medications("Гидроксихлорохин 200 мг по 1 таблетке 2 раза/сутки 2 недели")
    m = meds[0]
    assert m.dose_value == 200.0
    assert m.dose_unit == "мг"
    assert m.frequency and "2" in m.frequency
    assert m.duration and "недел" in m.duration.lower()


def test_trimedat_duration_days():
    meds = parse_medications("Тримедат форте 1 т 2 раза в день 28 дней")
    m = meds[0]
    assert m.duration and "28" in m.duration
    assert m.drug_name and "тримедат" in m.drug_name.lower()


def test_prednisolone_schedule():
    text = (
        "Преднизолон 5 мг по 12 таб в сутки\n"
        "С 12.08.24 - Преднизолон 5 мг по 11 таб в сутки\n"
        "С 19.08.24 - Преднизолон 5 мг по 10 таб в сутки"
    )
    meds = parse_medications(text)
    # все шаги одного препарата собираются в один MedicationItem со schedule
    pred = next(m for m in meds if m.drug_name and "преднизолон" in m.drug_name.lower())
    assert len(pred.schedule) >= 2
    starts = [s.start_date for s in pred.schedule if s.start_date]
    assert dt.date(2024, 8, 12) in starts


def test_non_drug_recommendations_skipped():
    text = "\n".join([
        "1. таб. аркоксия 90мг №7 утром",
        "4. консультация физиотерапевта",
        "5. ручной классический массаж поясничная область №10",
        "7. контрольная явка 23.11.2021",
        "л/н № АЮ 2784681",
    ])
    meds = parse_medications(text)
    raws = [m.raw_text for m in meds]
    assert any("аркоксия" in (r or "").lower() for r in raws)
    assert not any("массаж" in (r or "").lower() for r in raws)
    assert not any("контрольная явка" in (r or "").lower() for r in raws)
    assert all(looks_like_medication_item(m) for m in meds)
