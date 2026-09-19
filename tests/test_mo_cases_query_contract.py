"""Контракт фильтров списка МО: overall_grade и icd реально режут выборку."""
from __future__ import annotations

import inspect

from clinical_knowledge.mo_backend import _filter_records
from rag_server import api_methodist_mo_cases, api_methodist_mo_drugs_labs_kpis


def _row(case_id: str, **fields: object) -> dict:
    rec = {
        "case_id": case_id,
        "visit_id": case_id,
        "date": "2026-09-01",
        "document_kind": "clinical_visit",
        "zone1_band": "ok",
        "zone2a_band": "ok",
        "zone2b_band": "na",
        "zone2b_kp_status": "unmatched",
        "attention_primary": "none",
        "diagnosis_code": "",
        "mkb_code_main": "",
    }
    rec.update(fields)
    return rec


def test_cases_endpoint_declares_overall_grade_and_icd() -> None:
    names = set(inspect.signature(api_methodist_mo_cases).parameters)
    assert "overall_grade" in names
    assert "icd_visit_status" in names
    assert "icd" in names
    assert "queue_band" in names
    kpi_names = set(inspect.signature(api_methodist_mo_drugs_labs_kpis).parameters)
    assert "overall_grade" in kpi_names
    assert "finding_family" in kpi_names


def test_filter_records_overall_grade_good_vs_important() -> None:
    good = _row("g")
    important = _row("i", zone2a_band="bad")
    kept_good = _filter_records([good, important], {"overall_grade": "good"})
    kept_important = _filter_records([good, important], {"overall_grade": "important"})
    assert [row["case_id"] for row in kept_good] == ["g"]
    assert [row["case_id"] for row in kept_important] == ["i"]


def test_filter_records_overall_grade_pipe_multi() -> None:
    good = _row("g")
    important = _row("i", zone2a_band="bad")
    kept = _filter_records([good, important], {"overall_grade": "good|important"})
    assert {row["case_id"] for row in kept} == {"g", "i"}


def test_filter_records_icd_prefix() -> None:
    i10 = _row("a", diagnosis_code="I10", mkb_code_main="I10")
    j06 = _row("b", diagnosis_code="J06.9", mkb_code_main="J06.9")
    assert [row["case_id"] for row in _filter_records([i10, j06], {"icd": "I10"})] == ["a"]
    assert [row["case_id"] for row in _filter_records([i10, j06], {"icd": "J06"})] == ["b"]


def test_filter_records_queue_band_critical_not_overall_grade() -> None:
    critical = _row(
        "c",
        finding_codes=["C_red_flag"],
        _findings=[{"finding_code": "C_red_flag", "severity": "P0"}],
    )
    important = _row(
        "i",
        finding_codes=["B_dx_no_support"],
        _findings=[{"finding_code": "B_dx_no_support", "severity": "P1"}],
    )
    kept = _filter_records([critical, important], {"queue_band": "critical"})
    assert [row["case_id"] for row in kept] == ["c"]
    kept_imp = _filter_records([critical, important], {"queue_band": "important"})
    assert [row["case_id"] for row in kept_imp] == ["i"]
    by_visits = _filter_records(
        [critical, important],
        {"queue_band": "critical", "_queue_band_visits": {"c"}},
    )
    assert [row["case_id"] for row in by_visits] == ["c"]
