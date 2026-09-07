"""P1-2/P1-3 closeout: evaluated denominators, review projection, E04 HTTP, E23 verifier."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

import importlib.util

from clinical_knowledge.mo_backend import build_mo_drugs_labs_kpis
from clinical_knowledge.mo_daily import initialize_warehouse


def _snapshot(*, score: float = 61.0) -> str:
    return json.dumps(
        {
            "evaluation_run_id": "run-e04",
            "document_revision": 7,
            "source_hash": "hash-e04",
            "status": "completed",
            "value": score,
            "scores": {"overall_pct": score},
            "protocol": {"applicability_status": "not_evaluated"},
        },
        ensure_ascii=False,
    )


def _seed_parity_case(db: Path) -> None:
    initialize_warehouse(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO dim_doctor(doctor_key, doctor_fio, specialty) VALUES(?,?,?)",
            ("doc-e04", "Врач E04", "Терапия"),
        )
        conn.execute(
            """INSERT INTO fact_mo_case(
                 mis_id, visit_id, visit_date, document_kind, overall_pct, status,
                 doctor_key, specialty, filial, diagnosis_code, icd_chapter,
                 content_hash, source_hash, updated_at, evaluation_run_id,
                 document_revision, assessment_status, evaluation_snapshot_json
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "e04-1",
                "e04-1",
                "2026-09-01",
                "clinical_visit",
                61.0,
                "review",
                "doc-e04",
                "Терапия",
                "Центр",
                "J06.9",
                "X",
                "hash-e04",
                "hash-e04",
                "2026-09-01T00:00:00Z",
                "run-e04",
                7,
                "completed",
                _snapshot(),
            ),
        )
        conn.execute(
            """INSERT INTO fact_mo_finding
               (mis_id, finding_code, severity, passed, evidence, source_ref)
               VALUES (?,?,?,?,?,?)""",
            ("e04-1", "C_ddi", "P1", 0, "синтетика", "тест"),
        )
        conn.execute(
            """INSERT INTO crm_case_state(
                 case_id, status, assignee, tags_json, due_date,
                 finding_decisions_json, updated_at, updated_by
               ) VALUES (?,?,?,?,?,?,?,?)""",
            (
                "e04-1",
                "confirmed_issue",
                "Методист",
                "[]",
                "",
                json.dumps({"C_ddi": "confirmed"}, ensure_ascii=False),
                "2026-09-01T12:00:00Z",
                "Методист",
            ),
        )
        conn.commit()


def test_evaluated_denominator_and_review_projection(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    _seed_parity_case(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    payload = build_mo_drugs_labs_kpis({"date_from": "2026-09-01", "date_to": "2026-09-01"})
    row = payload["families"]["drug"]["by_doctor"][0]
    assert row["evaluated_cases"] == 1
    assert row["problem_pct_of_evaluated"] == 100.0
    assert row["ranking_eligible"] is False
    review = payload["families"]["drug"]["finding_provenance"]["review"]
    assert review["status"] == "projected"
    assert review["confirmed_cases"] == 1
    assert review["rejected_cases"] == 0


def test_e04_http_list_detail_export_parity(monkeypatch, tmp_path: Path) -> None:
    import rag_server

    db = tmp_path / "mo.sqlite"
    _seed_parity_case(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path / "empty"))
    monkeypatch.setenv("METHODIST_TOKEN", "synthetic-test-token")
    client = TestClient(
        rag_server.app,
        headers={"X-Methodist-Token": "synthetic-test-token", "X-Methodist-Role": "methodist"},
    )
    params = {"date_from": "2026-09-01", "date_to": "2026-09-01"}
    listed = client.get("/api/methodist/mo/cases", params=params)
    assert listed.status_code == 200
    rows = listed.json()["rows"]
    assert len(rows) == 1
    list_assessment = rows[0]["assessment"]
    detail = client.get("/api/methodist/mo/cases/e04-1")
    assert detail.status_code == 200
    detail_assessment = detail.json()["assessment"]
    exported = client.post(
        "/api/methodist/mo/exports",
        json={"kind": "cases", "filters": params},
    )
    assert exported.status_code == 200
    download = client.get(exported.json()["download_url"])
    assert download.status_code == 200
    export_rows = download.json()["rows"]
    export_assessment = export_rows[0]["assessment"]
    for key in ("evaluation_run_id", "document_revision", "value", "status"):
        assert list_assessment[key] == detail_assessment[key] == export_assessment[key]
    assert list_assessment["evaluation_run_id"] == "run-e04"
    assert list_assessment["document_revision"] == 7
    assert list_assessment["value"] == 61.0


def test_ux_leftovers_are_named_by_task() -> None:
    root = Path(__file__).resolve().parents[1]
    methodist = (root / "frontend/web/methodist/mis-kz-quality.html").read_text(encoding="utf-8")
    expert = (root / "frontend/web/methodist/expert.html").read_text(encoding="utf-8")
    app = (root / "frontend/web/shared/mo-app.js").read_text(encoding="utf-8")
    for html in (methodist, expert):
        assert "Автоматические сигналы: требуют проверки" in html
        assert "Поиск по выборке случаев" in html
        assert 'id="queue-bulk-bar"' in html
        assert "Shadow: плохо" not in html
    nav = methodist.split('id="app-nav"')[1].split("</ul>")[0]
    assert "Справка" not in nav
    assert "Фильтр строк этой таблицы" in app
    assert "periodAbsoluteRange" in app
    assert "период ещё не закончен" in app
    assert "applyColumnVisibility" in app
    assert app.index("bindSortableHeaders") < app.rindex("applyColumnVisibility(queue ? \"queue\" : \"documents\")")


def test_e23_runs_lab_image_verifier() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "deploy/gcp-app/verify_lab_assets.py"
    spec = importlib.util.spec_from_file_location("verify_lab_assets", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
