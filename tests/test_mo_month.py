from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import rag_server
from clinical_knowledge import mo_backend
from clinical_knowledge.mo_daily import doctor_key_for, initialize_warehouse


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "frontend" / "web" / "methodist" / "mis-kz-quality.html"
APP_PATH = ROOT / "frontend" / "web" / "shared" / "mo-app.js"


def _seed_month(path: Path) -> None:
    initialize_warehouse(path)
    doctors = [
        (doctor_key_for("Врач Альфа"), "Врач Альфа", "Терапия", "Центр"),
        (doctor_key_for("Врач Бета"), "Врач Бета", "Терапия", "Центр"),
        (doctor_key_for("Врач Гамма"), "Врач Гамма", "Кардиология", "Север"),
    ]
    now = "2026-07-04T05:00:00Z"
    with sqlite3.connect(path) as conn:
        conn.executemany("INSERT INTO dim_doctor VALUES (?,?,?,?)", doctors)
        for day_index, day in enumerate(("2026-07-01", "2026-07-02", "2026-07-03")):
            for index in range(20):
                global_index = day_index * 20 + index
                doctor = doctors[0] if global_index < 20 else doctors[1] if global_index < 40 else doctors[2]
                specialty = doctor[2]
                chapter = "IX" if specialty == "Терапия" else "X"
                score = 60.0 if doctor[1] == "Врач Альфа" else 80.0 if doctor[1] == "Врач Бета" else 90.0
                mis_id = f"jul-{global_index}"
                visit_id = f"visit-{global_index}"
                conn.execute(
                    """INSERT INTO fact_mo_case
                       (mis_id,visit_id,visit_date,document_kind,overall_pct,status,
                        doctor_key,specialty,filial,diagnosis_code,icd_chapter,content_hash,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        mis_id, visit_id, day, "clinical_visit", score, "needs_review" if score < 70 else "good",
                        doctor[0], specialty, doctor[3], "I10", chapter, f"hash-{global_index}", now,
                    ),
                )
                for axis, value in (
                    ("documentation", score),
                    ("clinical_concordance", score - 2),
                    ("safety", min(100, score + 5)),
                    ("regulatory", score + 1),
                ):
                    conn.execute(
                        "INSERT INTO fact_mo_score_axis(mis_id,axis,score) VALUES (?,?,?)",
                        (mis_id, axis, value),
                    )
                if global_index < 20:
                    code = "DOC_GAP" if global_index < 12 else "PLAN_GAP"
                    conn.execute(
                        """INSERT INTO fact_mo_finding
                           (mis_id,finding_code,severity,passed,evidence,source_ref)
                           VALUES (?,?,?,?,?,?)""",
                        (mis_id, code, "P1", 0, "private patient evidence", "private source"),
                    )
                if global_index < 5:
                    conn.execute(
                        """INSERT INTO crm_case_state
                           (case_id,status,assignee,tags_json,due_date,finding_decisions_json,updated_at,updated_by)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (visit_id, "in_review", "Методист", "[]", None, "{}", now, "Методист"),
                    )
                elif global_index < 10:
                    conn.execute(
                        """INSERT INTO crm_case_state
                           (case_id,status,assignee,tags_json,due_date,finding_decisions_json,updated_at,updated_by)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (visit_id, "closed", "Методист", "[]", None, "{}", now, "Методист"),
                    )
            conn.execute(
                """INSERT INTO fact_mo_daily
                   (visit_date,source_rows,scored_rows,avg_score,revision,quality_status,updated_at,
                    eligible_rows,partial,coverage_pct,avg_documentation,
                    avg_clinical_concordance,avg_safety,avg_regulatory,needs_attention,critical)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (day, 20, 20, 76.67, 1, "passed", now, 20, 0, 100.0, 76.67, 74.67, 81.67, 77.67, 20, 0),
            )
        for index in range(30):
            conn.execute(
                """INSERT INTO fact_mo_case
                   (mis_id,visit_id,visit_date,document_kind,overall_pct,status,
                    doctor_key,specialty,filial,diagnosis_code,icd_chapter,content_hash,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    f"jun-{index}", f"jun-visit-{index}", f"2026-06-{index // 10 + 1:02d}",
                    "clinical_visit", 70.0, "good", doctors[1][0], "Терапия", "Центр",
                    "I10", "IX", f"jun-hash-{index}", now,
                ),
            )


@pytest.fixture
def month_db(monkeypatch, tmp_path: Path) -> Path:
    path = tmp_path / "month.sqlite"
    _seed_month(path)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(path))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    return path


def test_month_contract_forecast_compare_and_reconciliation(month_db: Path) -> None:
    payload = mo_backend.build_month_report({"period": "month", "month": "2026-07"})

    assert payload["data_through"] == "2026-07-03"
    assert payload["days_elapsed"] == 3
    assert payload["days_in_month"] == 31
    assert payload["kpi"]["source_records"] == 60
    assert payload["kpi"]["evaluated"] == 60
    assert payload["forecast"]["projected_source"] == 620
    previous = payload["comparison"]["previous_month_equal_length"]
    assert previous["available"] is True
    assert previous["period"] == {"date_from": "2026-06-01", "date_to": "2026-06-03"}
    assert previous["kpi"]["source_records"] == 30
    assert payload["comparison"]["previous_year_same_period"]["available"] is False
    assert payload["reconciliation"]["status"] == "ok"
    assert payload["reconciliation"]["daily_source_sum"] == payload["kpi"]["source_records"]
    assert payload["reconciliation"]["daily_evaluated_sum"] == payload["kpi"]["evaluated"]


def test_month_contract_analytics_privacy_and_unavailable_reg55(month_db: Path) -> None:
    payload = mo_backend.build_month_report({"period": "month", "month": "2026-07"})

    assert len(payload["timeseries"]["items"]) == 3
    assert all("documentation" in item and "volume" in item for item in payload["timeseries"]["items"])
    assert payload["heatmap"]["cells"]
    assert any(item["enough_data"] for item in payload["doctor_case_mix"]["items"])
    assert payload["pareto"]["items"][0]["cumulative_share_pct"] < 100
    assert payload["pareto"]["items"][-1]["cumulative_share_pct"] == 100
    assert payload["funnel"] == {
        "source": 60, "eligible": 60, "evaluated": 60,
        "with_findings": 20, "in_crm_work": 5, "closed": 5,
    }
    assert payload["crm_progress"]["statuses"]["in_review"] == 5
    assert payload["reg55"]["available"] is False
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "private patient evidence" not in serialized
    assert "private source" not in serialized
    assert "patient_id" not in serialized


def test_month_reconciliation_exposes_divergence(month_db: Path) -> None:
    with sqlite3.connect(month_db) as conn:
        conn.execute("UPDATE fact_mo_daily SET source_rows=19 WHERE visit_date='2026-07-03'")
    payload = mo_backend.build_month_report({"period": "month", "month": "2026-07"})
    assert payload["reconciliation"]["status"] == "diverged"
    assert payload["reconciliation"]["source_delta"] == -1


def test_month_http_requires_auth_and_is_private(monkeypatch, month_db: Path) -> None:
    monkeypatch.setenv("METHODIST_TOKEN", "month-token")
    client = TestClient(rag_server.app)
    url = "/api/methodist/mo/month-report?period=month&month=2026-07"
    assert client.get(url).status_code == 403
    response = client.get(url, headers={"X-Methodist-Token": "month-token"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json()["data_through"] == "2026-07-03"


def test_month_markup_and_javascript_source() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    app = APP_PATH.read_text(encoding="utf-8")
    for marker in (
        'id="month-kpis"', 'id="month-forecast"', 'id="month-trend-chart"',
        'id="month-heatmap-chart"', 'id="month-doctor-chart"', 'id="month-pareto-chart"',
        'id="month-funnel-chart"', 'id="month-crm-chart"', 'id="month-reg55"',
    ):
        assert marker in html
    assert 'request("/month-report"' in app
    assert "renderMonthTrend" in app
    assert "renderMonthHeatmap" in app
    assert "renderMonthDoctors" in app
    assert "renderMonthPareto" in app
    assert app.count('MO.moChart($("month-') >= 6
    assert "dataZoom" in app


def test_month_javascript_syntax() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    result = subprocess.run(
        [node, "--check", str(APP_PATH)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
