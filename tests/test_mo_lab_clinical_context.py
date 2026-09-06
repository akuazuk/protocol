"""Numerical lab context reaches the evaluator without display truncation or future data."""
import sqlite3

import pytest

from clinical_knowledge import mo_lab_bundle
from clinical_knowledge.lab_abnormal_findings import CODE_ABNORMAL_IGNORED, _parse_number, _unit_ok
from clinical_knowledge.mo_lab_shadow import evaluate_lab_for_case


@pytest.fixture
def lab(monkeypatch):
    monkeypatch.setenv("MO_LAB_BUNDLE", "1")
    monkeypatch.setenv("MO_LAB_ABNORMAL", "1")
    monkeypatch.setenv("MO_LAB_ABNORMAL_PRIMARY", "0")
    db = sqlite3.connect(":memory:")
    db.execute("""CREATE TABLE fact_mo_lab(
        patient_key TEXT, test_date TEXT, test_id INTEGER, type_id INTEGER,
        type_name TEXT, indicator_id INTEGER, indicator_name TEXT, value TEXT, unit TEXT)""")
    yield db
    db.close()


def insert(lab, day="2026-08-19", value="12.5", unit="ммоль/л", key="synthetic-patient"):
    lab.execute("INSERT INTO fact_mo_lab VALUES(?,?,1,1,'Synthetic panel',1,'Глюкоза',?,?)", (key, day, value, unit))


def evaluate(lab, **overrides):
    case = {"patient_key": "synthetic-patient", "visit_date": "2026-08-20", "age_years": 40, **overrides}
    return evaluate_lab_for_case(case, lab_db=lab)


def abnormal_codes(findings):
    return [f["code"] for f in findings if f["code"] == CODE_ABNORMAL_IGNORED]


def test_full_entrypoint_uses_values_not_reconcile_projection(lab):
    insert(lab)
    payload, findings = evaluate(lab)
    assert abnormal_codes(findings) == [CODE_ABNORMAL_IGNORED]
    assert next(f for f in findings if f["code"] == CODE_ABNORMAL_IGNORED)["shadow"] is True
    assert payload["abnormal_check"]["date_precision"] == "day"
    bundle = mo_lab_bundle.build_lab_reconcile_bundle(patient_key="synthetic-patient", visit_date="2026-08-20", lab_db=lab)
    assert bundle["days"][0]["types"][0]["indicators"][0]["value"] == ""
    assert "synthetic-patient" not in str(payload)


def test_display_cap_cannot_hide_numerical_evidence(lab, monkeypatch):
    monkeypatch.setattr(mo_lab_bundle, "ROW_CAP", 1)
    insert(lab, day="2026-08-19", value="12.5")
    insert(lab, day="2026-08-20", value="5.0")
    payload, findings = evaluate(lab)
    assert payload["summary"]["truncated"] is True
    assert abnormal_codes(findings) == [CODE_ABNORMAL_IGNORED]


@pytest.mark.parametrize("kwargs", [
    {"day": "2026-08-21"},
    {"key": "synthetic-other"},
    {"unit": ""},
    {"unit": "моль/л"},
    {"value": "<12.5"},
    {"value": "5.0"},
])
def test_unavailable_or_unevaluable_result_does_not_accuse(lab, kwargs):
    insert(lab, **kwargs)
    _, findings = evaluate(lab)
    assert not abnormal_codes(findings)


@pytest.mark.parametrize("age", [None, 7, 17])
def test_adult_seed_is_not_used_for_unknown_age_or_children(lab, age):
    insert(lab)
    payload, findings = evaluate(lab, age_years=age)
    assert not abnormal_codes(findings)
    assert payload["abnormal_check"]["status"] in {"not_evaluated", "not_applicable"}


def test_unit_aliases_preserve_scale():
    assert _unit_ok("mmol/l", "ммоль/л")
    assert not _unit_ok("мме/л", "ме/л")
    assert not _unit_ok("моль/л", "ммоль/л")
    assert not _unit_ok("", "ммоль/л")
    assert not _unit_ok("ммоль/л", "")


def test_numbers_are_exact_not_fragments():
    assert _parse_number(0) == 0
    assert _parse_number("12,5") == 12.5
    for value in ["<12.5", ">5", "12-15", "не определено", "5 мг"]:
        assert _parse_number(value) is None
