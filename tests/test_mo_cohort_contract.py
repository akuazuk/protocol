"""Calendar and population parity through the real MO HTTP routes (synthetic data)."""
from datetime import date
import sqlite3

import pytest
from fastapi.testclient import TestClient

from clinical_knowledge import mo_metrics
from clinical_knowledge.mo_daily import initialize_warehouse


@pytest.fixture
def client(monkeypatch, tmp_path):
    import rag_server

    db = tmp_path / "warehouse.sqlite"
    initialize_warehouse(db)
    # Adjacent months, eligible legacy records and excluded document/status/branch.
    rows = [
        ("synthetic-a", "2026-07-31", "clinical_visit", "review", "North"),
        ("synthetic-b", "2026-08-01", "clinical_visit", "review", "North"),
        ("synthetic-c", "2026-08-02", "clinical_visit", "review", "North"),
        ("synthetic-d", "2026-08-02", "consultation", "review", "North"),
        ("synthetic-e", "2026-08-02", "procedure_session", "review", "North"),
        ("synthetic-f", "2026-08-02", "clinical_visit", "good", "South"),
        ("synthetic-g", "2026-08-03", "clinical_visit", "review", "North"),
    ]
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO dim_doctor(doctor_key,doctor_fio) VALUES('synthetic-doctor','Synthetic doctor')")
        for key, day, kind, status, branch in rows:
            conn.execute(
                """INSERT INTO fact_mo_case
                (mis_id,visit_id,visit_date,document_kind,status,filial,doctor_key,specialty,
                 overall_pct,content_hash,updated_at) VALUES(?,?,?,?,?,?,'synthetic-doctor','Synthetic specialty',60,'synthetic',?)""",
                (key, key, day, kind, status, branch, day),
            )
            conn.execute(
                "INSERT INTO fact_mo_finding(mis_id,finding_code,severity,passed) VALUES(?,'C_ddi','P1',0)",
                (key,),
            )
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path / "no-source"))
    monkeypatch.setenv("METHODIST_TOKEN", "synthetic-test-token")
    monkeypatch.setattr(mo_metrics, "minsk_today", lambda now=None: date(2026, 8, 3))
    return TestClient(rag_server.app, headers={"X-Methodist-Token": "synthetic-test-token"})


@pytest.mark.parametrize("params, expected", [
    ({"period": "month", "month": "2026-07"}, 1),
    ({"period": "month", "month": "2026-08"}, 3),
    ({"period": "yesterday"}, 2),
    ({"period": "7d"}, 4),
    ({"period": "custom", "date_from": "2026-08-01", "date_to": "2026-08-01"}, 1),
    ({"period": "month", "month": "2026-08", "statuses": "review", "filials": "North"}, 2),
    ({"period": "month", "month": "2026-08", "doctors": "Absent"}, 0),
    ({"period": "month", "month": "2026-06"}, 0),
])
def test_kpi_list_calendar_and_filters_agree(client, params, expected):
    params = {**params, "document_kinds": "clinical_visit"}
    kpi = client.get("/api/methodist/mo/drugs-labs-kpis", params=params)
    cases = client.get("/api/methodist/mo/cases", params=params)
    assert kpi.status_code == cases.status_code == 200
    payload = kpi.json()
    assert payload["ok"] is True
    assert cases.json()["total"] == payload["denominators"]["total_cases"] == expected
    assert payload["families"]["drug"]["cases"] == expected
    assert payload["date_from"] and payload["date_to"]


def test_explicit_document_selection_intersects_eligibility(client):
    params = {"period": "month", "month": "2026-08"}
    payload = client.get("/api/methodist/mo/drugs-labs-kpis", params=params).json()
    assert payload["denominators"]["total_cases"] == 4
    payload = client.get("/api/methodist/mo/drugs-labs-kpis", params={**params, "document_kinds": "procedure_session"}).json()
    assert payload["denominators"]["total_cases"] == 0


@pytest.mark.parametrize("endpoint", ["cases", "facets", "overview", "drugs-labs-kpis"])
def test_invalid_period_is_rejected_not_silently_ignored(client, endpoint):
    response = client.get("/api/methodist/mo/" + endpoint, params={"period": "quarter"})
    assert response.status_code == 422


def test_month_overrides_stale_custom_dates(client):
    params = {"period": "month", "month": "2026-07", "date_from": "2026-08-01", "date_to": "2026-08-02", "document_kinds": "clinical_visit"}
    kpi = client.get("/api/methodist/mo/drugs-labs-kpis", params=params).json()
    cases = client.get("/api/methodist/mo/cases", params=params).json()
    assert kpi["date_from"] == "2026-07-01"
    assert kpi["date_to"] == "2026-07-31"
    assert kpi["denominators"]["total_cases"] == cases["total"] == 1
