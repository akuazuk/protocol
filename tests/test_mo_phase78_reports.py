"""Контрактные тесты фаз 7-8: health, документ МО, отчёты без «пустых» KPI."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

import rag_server
from clinical_knowledge.mo_daily import doctor_key_for, initialize_warehouse
from clinical_knowledge.mo_case_document import render_case_document_html, score_reason
from clinical_knowledge.mo_report_engine import build_telegram_briefing


def _seed(path: Path) -> str:
    initialize_warehouse(path)
    doctor = doctor_key_for("Тестовый Врач")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO dim_doctor(doctor_key,doctor_fio,specialty,filial) VALUES(?,?,?,?)",
            (doctor, "Тестовый Врач", "Терапия", "Центр"),
        )
        conn.execute(
            """INSERT INTO fact_mo_case
               (mis_id,visit_id,visit_date,document_kind,overall_pct,status,
                doctor_key,specialty,filial,diagnosis_code,icd_chapter,content_hash,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "9001",
                "91001",
                "2026-08-01",
                "consultation",
                48.0,
                "review",
                doctor,
                "Терапия",
                "Центр",
                "J35.0",
                "Болезни органов дыхания",
                "abc123hashshouldnotshow",
                "2026-08-02T00:00:00Z",
            ),
        )
        conn.execute(
            """INSERT INTO fact_mo_case
               (mis_id,visit_id,visit_date,document_kind,overall_pct,status,
                doctor_key,specialty,filial,diagnosis_code,icd_chapter,content_hash,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "9002",
                "91002",
                "2026-08-01",
                "diagnostic",
                None,
                "",
                doctor,
                "Терапия",
                "Центр",
                "N60",
                "Болезни мочеполовой системы",
                "b7c32c8625076ffe4c279ade14e19ed95bbec62936ada2367a80e0483efc8c8a",
                "2026-08-02T00:00:00Z",
            ),
        )
        conn.execute(
            """INSERT INTO fact_mo_finding
               (mis_id,finding_code,severity,passed,evidence,source_ref)
               VALUES(?,?,?,?,?,?)""",
            ("9001", "A_missing_recommendations", "P0", 0, "нет рекомендаций", "СОП №2"),
        )
        conn.execute(
            """INSERT INTO fact_mo_daily
               (visit_date,source_rows,scored_rows,avg_score,revision,quality_status,updated_at,
                eligible_rows,partial,coverage_pct,avg_documentation,avg_clinical_concordance,
                avg_safety,avg_regulatory,needs_attention,critical)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "2026-08-01",
                10,
                8,
                80.0,
                1,
                "passed",
                "2026-08-02T00:00:00Z",
                8,
                0,
                100.0,
                80.0,
                80.0,
                90.0,
                70.0,
                1,
                1,
            ),
        )
        conn.commit()
    return doctor


def test_score_reason_for_diagnostic() -> None:
    assert "Не оценивается" in score_reason(document_kind="diagnostic", overall_pct=None)


def test_health_reports_document_and_cabinet(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    doctor = _seed(db)
    monkeypatch.setenv("METHODIST_TOKEN", "test-token")
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    headers = {"X-Methodist-Token": "test-token", "X-Methodist-Role": "methodist"}
    client = TestClient(rag_server.app)

    health = client.get("/api/methodist/mo/health", headers=headers)
    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert health.json()["features"]["case_document"] is True

    reports = client.get("/api/methodist/mo/reports", headers=headers)
    assert reports.status_code == 200
    items = reports.json()["items"]
    assert items
    by_date = {item["date"]: item for item in items if item.get("date")}
    assert "2026-08-01" in by_date
    assert by_date["2026-08-01"]["evaluated"] == 8
    assert by_date["2026-08-01"]["critical"] == 1

    doc = client.get("/api/methodist/mo/cases/91001/document", headers=headers)
    assert doc.status_code == 200
    assert "text/html" in doc.headers.get("content-type", "")
    assert "Тестовый Врач" in doc.text
    assert "J35.0" in doc.text
    assert "abc123hashshouldnotshow" not in doc.text

    cabinet = client.get(
        f"/api/methodist/mo/doctor-cabinet?doctor_key={doctor}",
        headers=headers,
    )
    assert cabinet.status_code == 200
    payload = cabinet.json()
    assert payload["ok"] is True
    assert len(payload["cases"]) == 1
    assert payload["hidden_unscored"] == 1
    assert payload["cases"][0]["diagnosis_code"] == "J35.0"
    assert "b7c32c" not in json.dumps(payload, ensure_ascii=False)

    unscored = client.get(
        f"/api/methodist/mo/doctor-cabinet?doctor_key={doctor}&include_unscored=true",
        headers=headers,
    ).json()
    assert len(unscored["cases"]) == 2
    diagnostic = next(item for item in unscored["cases"] if item["document_kind"] == "diagnostic")
    assert "Не оценивается" in (diagnostic.get("score_reason") or "")


def test_telegram_briefing_contains_links() -> None:
    text = build_telegram_briefing(
        {
            "date": "2026-08-01",
            "summary": {"avg_score": 88, "scored": 10, "source_rows": 12, "needs_attention": 2, "critical": 1},
            "action_queue": [
                {
                    "priority": "P0",
                    "doctor_fio": "Иванов",
                    "score": 40,
                    "visit_id": "91001",
                    "mis_id": "9001",
                }
            ],
        }
    )
    assert "91001/pdf" in text
    assert "Критические: 1" in text


def test_render_case_html_never_shows_content_hash() -> None:
    html = render_case_document_html(
        {
            "visit_date": "2026-08-01",
            "visit_id": "91001",
            "mis_id": "9001",
            "doctor_fio": "Врач",
            "specialty": "Терапия",
            "filial": "Центр",
            "document_kind_label": "Консультативное заключение",
            "diagnosis_code": "J35.0",
            "overall_pct": 48,
            "score_reason": "Оценено",
            "axes": {},
            "findings": [{"severity": "P0", "finding_code": "A_x", "title_ru": "Нет рекомендаций"}],
            "clinical": {"clinical_diagnosis": "Хронический тонзиллит"},
            "case_id": "91001",
            "generated_at": "2026-08-03T00:00:00Z",
        }
    )
    assert "Хронический тонзиллит" in html
    assert "b7c32c8625076ffe" not in html
