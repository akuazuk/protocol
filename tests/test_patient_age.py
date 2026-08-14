from __future__ import annotations

from datetime import date

from clinical_knowledge.patient_age import age_years_on, resolve_patient_age


def test_age_years_on_before_birthday() -> None:
    born = date(2019, 8, 20)
    assert age_years_on(born, date(2026, 7, 15)) == 6
    assert age_years_on(born, date(2026, 8, 20)) == 7


def test_resolve_prefers_ready_years() -> None:
    meta = resolve_patient_age(
        {"patient_age_years": 7, "patient_bdate": "2000-01-01", "visit_date": "2026-07-15"},
        {},
    )
    assert meta["age_years"] == 7
    assert meta["audience"] == "child"
    assert meta["age_source"] == "age_years"


def test_resolve_bdate_plus_visit_not_today() -> None:
    meta = resolve_patient_age(
        {"patient_bdate": "2019-03-01", "visit_date": "2026-07-15"},
        {},
        today=date(2030, 1, 1),
    )
    assert meta["age_years"] == 7
    assert meta["audience"] == "child"
    assert meta["age_source"] == "bdate_visit"
    assert meta["visit_date"] == "2026-07-15"


def test_resolve_adult_from_years() -> None:
    meta = resolve_patient_age({"patient_age_years": 41}, {"visit_date": "2026-07-10"})
    assert meta["audience"] == "adult"
    assert meta["age_years"] == 41


def test_resolve_unknown_without_age_or_bdate() -> None:
    meta = resolve_patient_age({"visit_date": "2026-07-15"}, {})
    assert meta["audience"] == "unknown"
    assert meta["age_years"] is None
