"""F2: /cases/summary - тот же WHERE, что /cases: оценки, топ-8 специальностей, недели."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from clinical_knowledge import mo_backend
from rag_server import api_methodist_mo_cases, api_methodist_mo_cases_summary
from test_mo_overview_dashboard_f1 import BASE, _seed


@pytest.fixture()
def warehouse(monkeypatch, tmp_path: Path) -> Path:
    db = tmp_path / "cases_summary.sqlite"
    _seed(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_RESULT_CACHE", "0")
    monkeypatch.setattr(mo_backend, "_WAREHOUSE_SCHEMA_READY_PATH", None)
    return db


PAGING = {"page", "page_size", "sort_by", "sort_dir"}


def test_summary_declares_same_filters_as_cases_minus_paging() -> None:
    cases = set(inspect.signature(api_methodist_mo_cases).parameters)
    summary = set(inspect.signature(api_methodist_mo_cases_summary).parameters)
    assert PAGING <= cases
    assert PAGING.isdisjoint(summary)
    assert (cases - {"request"} - PAGING) <= (summary - {"request", "response"})


def test_summary_matches_cases_total_and_has_three_blocks(warehouse: Path) -> None:
    params = {**BASE, "document_kinds": "clinical_visit", "score_eligible_only": "1"}
    summary = mo_backend.build_cases_summary(dict(params))
    cases = mo_backend.build_cases({**params, "page_size": 200})
    assert summary["ok"] and summary["available"]
    assert summary["n"] == cases["total"] == 40
    assert summary["n"] == sum(summary["grades"]["totals"].values())
    assert [b["id"] for b in summary["grades"]["buckets"]] == list(mo_backend._CASES_SUMMARY_GRADE_KEYS)
    assert summary["grades"]["totals"]["na"] == 4
    assert summary["grades"]["totals"]["critical"] == 1
    specs = {row["value"]: row["n"] for row in summary["specialties"]}
    assert specs == {"Терапия": 20, "Кардиология": 20}
    assert summary["weeks"]
    assert all(row["week"].startswith("2026-W") for row in summary["weeks"])
    assert sum(row["n"] for row in summary["weeks"]) == 40


def test_summary_filters_match_cases(warehouse: Path) -> None:
    base = {**BASE, "document_kinds": "clinical_visit", "score_eligible_only": "1"}
    spec = mo_backend.build_cases_summary({**base, "specializations": "Терапия"})
    spec_cases = mo_backend.build_cases({**base, "specializations": "Терапия", "page_size": 200})
    assert spec["n"] == spec_cases["total"] == 20
    assert [row["value"] for row in spec["specialties"]] == ["Терапия"]

    good = mo_backend.build_cases_summary({**base, "overall_grade": "good"})
    good_cases = mo_backend.build_cases({**base, "overall_grade": "good", "page_size": 200})
    assert good["n"] == good_cases["total"]
    assert good["n"] > 0
    assert good["grades"]["totals"]["good"] == good["n"]
    assert sum(v for k, v in good["grades"]["totals"].items() if k != "good") == 0

    na = mo_backend.build_cases_summary({**base, "overall_grade": "na"})
    na_cases = mo_backend.build_cases({**base, "overall_grade": "na", "page_size": 200})
    assert na["n"] == na_cases["total"] == 4


def test_summary_accepts_text_search_without_crashing(warehouse: Path) -> None:
    empty = mo_backend.build_cases_summary({**BASE, "document_kinds": "clinical_visit"})
    assert not (empty.get("search_plan") or {}).get("chips")
    with_q = mo_backend.build_cases_summary({**BASE, "document_kinds": "clinical_visit", "q": "I10"})
    assert with_q["ok"]
    assert with_q["n"] <= empty["n"]
