"""F6: /queue-dashboard - плитки CRM, возраст задачи, ответственные."""
from __future__ import annotations

import inspect
from datetime import date, timedelta
from pathlib import Path

import pytest

from clinical_knowledge import mo_backend
from clinical_knowledge.mo_daily import doctor_key_for, initialize_warehouse
from rag_server import api_methodist_mo_queue_dashboard


def _seed_queue(path: Path) -> None:
    initialize_warehouse(path)
    today = date.today()
    yesterday = today - timedelta(days=1)
    doc = doctor_key_for("Врач А")
    rows = [
        ("q01", "201", yesterday.isoformat(), "new", "", None),
        ("q02", "202", yesterday.isoformat(), "new", "", None),
        ("q03", "203", (today - timedelta(days=3)).isoformat(), "in_review", "метод А", None),
        ("q04", "204", (today - timedelta(days=5)).isoformat(), "confirmed_issue", "метод А", None),
        ("q05", "205", (today - timedelta(days=12)).isoformat(), "assigned", "метод Б", None),
        ("q06", "206", (today - timedelta(days=20)).isoformat(), "closed", "", None),
    ]
    with __import__("sqlite3").connect(path) as conn:
        conn.execute(
            "INSERT INTO dim_doctor(doctor_key,doctor_fio,specialty,filial) VALUES(?,?,?,?)",
            (doc, "Врач А", "Терапевт", "Центр"),
        )
        for mis_id, visit_id, visit_date, status, assignee, _due in rows:
            conn.execute(
                """INSERT INTO fact_mo_case
                   (mis_id,visit_id,visit_date,document_kind,overall_pct,doctor_key,specialty,filial,
                    diagnosis_code,content_hash,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mis_id,
                    visit_id,
                    visit_date,
                    "clinical_visit",
                    55.0,
                    doc,
                    "Терапевт",
                    "Центр",
                    "I10",
                    mis_id,
                    today.isoformat() + "T00:00:00Z",
                ),
            )
            conn.execute(
                "INSERT INTO fact_mo_finding(mis_id,finding_code,severity,passed,title_ru) VALUES(?,?,?,?,?)",
                (mis_id, "B_dx_not_justified", "P1", 0, "Диагноз не обоснован"),
            )
            conn.execute(
                """INSERT INTO crm_case_state(case_id,status,assignee,tags_json,due_date,finding_decisions_json,updated_at,updated_by)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (visit_id, status, assignee or None, "[]", None, "{}", today.isoformat() + "T00:00:00Z", "t"),
            )
        conn.commit()


@pytest.fixture()
def warehouse(monkeypatch, tmp_path: Path) -> Path:
    db = tmp_path / "queue_dash.sqlite"
    _seed_queue(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    mo_backend._result_cache_clear()
    return db


def test_queue_dashboard_excludes_closed_and_buckets_age(warehouse: Path) -> None:
    dash = mo_backend.build_queue_dashboard(
        {
            "period": "custom",
            "date_from": (date.today() - timedelta(days=40)).isoformat(),
            "date_to": (date.today() - timedelta(days=1)).isoformat(),
        }
    )
    assert dash["ok"] is True
    assert dash["available"] is True
    assert dash["total"] == 5
    tiles = {row["id"]: row["n"] for row in dash["tiles"]}
    assert tiles["new"] == 2
    assert tiles["in_review"] == 1
    assert tiles["confirmed_issue"] == 1
    assert tiles["assigned"] == 1
    ages = {row["id"]: row["n"] for row in dash["ages"]}
    assert ages["0-1"] == 2
    assert ages["2-3"] == 1
    assert ages["4-7"] == 1
    assert ages[">7"] == 1
    owners = {row["label"]: row["n"] for row in dash["owners"]}
    assert owners["метод А"] == 2
    assert owners["метод Б"] == 1
    assert owners["не назначен"] == 2


def test_queue_dashboard_route_registered_before_case_id() -> None:
    source = inspect.getsource(api_methodist_mo_queue_dashboard)
    assert "build_queue_dashboard" in source
    text = Path(__file__).resolve().parents[1].joinpath("rag_server.py").read_text(encoding="utf-8")
    assert text.index("/api/methodist/mo/queue-dashboard") < text.index(
        '/api/methodist/mo/cases/{case_id}/document'
    )
