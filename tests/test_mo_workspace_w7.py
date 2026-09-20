"""W7: overall_grade in SQL + warehouse indexes."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from clinical_knowledge.mo_backend import (
    _cases_sql_pageable,
    _sql_overall_grade_expr,
    _warehouse_where,
)
from clinical_knowledge.mo_overall_grade import attach_overall_grade

ROOT = Path(__file__).resolve().parents[1]
DAILY = (ROOT / "clinical_knowledge" / "mo_daily.py").read_text(encoding="utf-8")
SCRIPT = (ROOT / "scripts" / "ensure_mo_warehouse_indexes.py").read_text(encoding="utf-8")

SAMPLES = [
    {
        "attention_primary": "safety",
        "zone1_band": "ok",
        "zone2a_band": "ok",
        "zone2b_band": "ok",
        "zone2b_kp_status": "matched",
    },
    {
        "attention_primary": "",
        "zone1_band": "ok",
        "zone2a_band": "bad",
        "zone2b_band": "ok",
        "zone2b_kp_status": "matched",
    },
    {
        "attention_primary": "",
        "zone1_band": "ok",
        "zone2a_band": "ok",
        "zone2b_band": "bad",
        "zone2b_kp_status": "matched",
    },
    {
        "attention_primary": "",
        "zone1_band": "bad",
        "zone2a_band": "ok",
        "zone2b_band": "ok",
        "zone2b_kp_status": "unmatched",
    },
    {
        "attention_primary": "",
        "zone1_band": "weak",
        "zone2a_band": "ok",
        "zone2b_band": "ok",
        "zone2b_kp_status": "unmatched",
    },
    {
        "attention_primary": "",
        "zone1_band": "ok",
        "zone2a_band": "ok",
        "zone2b_band": "ok",
        "zone2b_kp_status": "matched",
    },
    {
        "attention_primary": "",
        "zone1_band": "ok",
        "zone2a_band": "ok",
        "zone2b_band": "bad",
        "zone2b_kp_status": "unmatched",
    },
]


def test_sql_overall_grade_matches_python() -> None:
    expr = _sql_overall_grade_expr("t")
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE t (attention_primary TEXT, zone1_band TEXT, zone2a_band TEXT, "
        "zone2b_band TEXT, zone2b_kp_status TEXT)"
    )
    for row in SAMPLES:
        conn.execute(
            "INSERT INTO t VALUES (?,?,?,?,?)",
            (
                row["attention_primary"],
                row["zone1_band"],
                row["zone2a_band"],
                row["zone2b_band"],
                row["zone2b_kp_status"],
            ),
        )
        sql_grade = conn.execute(f"SELECT {expr} FROM t").fetchone()[0]
        py_grade = attach_overall_grade(dict(row))["overall_grade"]["grade"]
        assert sql_grade == py_grade, (row, sql_grade, py_grade)
        conn.execute("DELETE FROM t")


def test_warehouse_where_filters_overall_grade() -> None:
    where, values = _warehouse_where(
        {
            "date_from": "2026-09-01",
            "date_to": "2026-09-20",
            "overall_grade": "poor|important|critical",
        }
    )
    joined = " AND ".join(where)
    assert "zone2a_band" in joined
    assert "important" in values
    assert "poor" in values
    assert "critical" in values


def test_overall_grade_is_sql_pageable(monkeypatch) -> None:
    monkeypatch.setattr("clinical_knowledge.mo_backend._backend_source", lambda: "warehouse")
    assert _cases_sql_pageable({"overall_grade": "good", "date_from": "2026-09-01"})


def test_indexes_are_declared() -> None:
    assert "idx_finding_mis ON fact_mo_finding(mis_id)" in DAILY
    assert "idx_case_zone_bands" in DAILY
    assert "idx_case_attention" in DAILY
    assert "idx_case_zone_bands" in SCRIPT
    assert "idx_fact_mo_case_visit" in SCRIPT
