"""W5: warehouse search by clinical diagnosis text, SQL LIKE, typeahead."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.mis_kz_quality import _match_filters
from clinical_knowledge.mo_backend import (
    _cases_sql_pageable,
    _describe_empty_state,
    _sql_like_contains,
    _warehouse_where,
)

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "web" / "methodist" / "mis-kz-quality.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
DAILY = (ROOT / "clinical_knowledge" / "mo_daily.py").read_text(encoding="utf-8")


def test_sql_like_escapes_wildcards() -> None:
    assert _sql_like_contains("100%") == r"%100\%%"
    assert _sql_like_contains("a_b") == r"%a\_b%"


def test_warehouse_q_searches_diagnosis_text() -> None:
    where, values = _warehouse_where(
        {"q": "гипертон", "date_from": "2026-09-01", "date_to": "2026-09-20"}
    )
    joined = " AND ".join(where)
    assert "c.diagnosis_text" in joined
    assert "dx.diagnosis_label" in joined
    assert any("гипертон" in str(v).lower() for v in values)


def test_q_is_sql_pageable(monkeypatch) -> None:
    monkeypatch.setattr("clinical_knowledge.mo_backend._backend_source", lambda: "warehouse")
    assert _cases_sql_pageable({"q": "гипертон", "date_from": "2026-09-01", "date_to": "2026-09-20"})


def test_empty_state_names_diagnosis() -> None:
    empty = _describe_empty_state(
        total_records=10,
        filtered_records=0,
        params={"q": "гипертон"},
    )
    assert empty["reason_code"] == "search_text_miss"
    assert empty["title"] == "Диагноз не найден"
    assert "клинический диагноз" in empty["hint"].lower()
    assert "не название болезни" not in empty["hint"].lower()


def test_python_filter_uses_diagnosis_text() -> None:
    rec = {
        "visit_id": "1",
        "case_id": "1",
        "mis_id": "1",
        "doctor_fio": "Иванов",
        "diagnosis_short": "Не указан",
        "diagnosis_text": "Гипертоническая болезнь",
        "mkb_code_main": "",
        "date": "2026-09-10",
    }
    assert _match_filters(rec, {"q": "гипертон"})
    assert not _match_filters(rec, {"q": "миозит"})


def test_ui_typeahead_and_placeholder() -> None:
    assert 'placeholder="Врач, диагноз, МКБ, visit_id"' in HTML
    assert "/dx-suggest?" in APP
    assert "Диагноз не найден" in APP
    assert "не название болезни" not in APP
    assert "idx_case_diagnosis_text" in DAILY
    assert "idx_fact_mo_case_visit" in DAILY
