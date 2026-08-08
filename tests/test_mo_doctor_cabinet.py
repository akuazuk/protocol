from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

import rag_server
from clinical_knowledge import mo_backend
from clinical_knowledge.mo_daily import (
    detect_template_copies,
    doctor_key_for,
    initialize_warehouse,
    upsert_warehouse,
)


def _seed(path: Path) -> tuple[str, str]:
    initialize_warehouse(path)
    doctor_a = doctor_key_for("Врач А")
    doctor_b = doctor_key_for("Врач Б")
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT INTO dim_doctor(doctor_key,doctor_fio,specialty,filial) VALUES(?,?,?,?)",
            [
                (doctor_a, "Врач А", "Терапия", "Центр"),
                (doctor_b, "Врач Б", "Хирургия", "Север"),
            ],
        )
        for index, doctor_key in enumerate((doctor_a, doctor_b), start=1):
            conn.execute(
                """INSERT INTO fact_mo_case
                   (mis_id,visit_id,visit_date,document_kind,overall_pct,status,
                    doctor_key,specialty,filial,diagnosis_code,icd_chapter,content_hash,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    f"case-{index}",
                    f"visit-{index}",
                    "2026-07-29",
                    "clinical_visit",
                    75.0,
                    "review",
                    doctor_key,
                    "Терапия" if index == 1 else "Хирургия",
                    "Центр",
                    "I10",
                    "Болезни системы кровообращения",
                    str(index),
                    "2026-07-30T00:00:00Z",
                ),
            )
            conn.execute(
                """INSERT INTO fact_mo_finding
                   (mis_id,finding_code,severity,passed,evidence,source_ref)
                   VALUES(?,?,?,?,?,?)""",
                (f"case-{index}", "B_gap", "P1", 0, "", "protocol:55:2"),
            )
        conn.execute(
            """INSERT INTO crm_case_state
               (case_id,status,assignee,tags_json,due_date,finding_decisions_json,updated_at,updated_by)
               VALUES(?,?,?,?,?,?,?,?)""",
            ("visit-1", "in_review", "Методист", '["важно"]', None, "{}", "2026-07-30T00:00:00Z", "Методист"),
        )
        conn.commit()
    return doctor_a, doctor_b


def test_doctor_identity_is_required_and_query_cannot_override(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    doctor_a, doctor_b = _seed(db)
    monkeypatch.setenv("METHODIST_TOKEN", "test-token")
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    client = TestClient(rag_server.app)
    doctor_headers = {
        "X-Methodist-Token": "test-token",
        "X-Methodist-Role": "doctor",
    }

    denied = client.get(
        f"/api/methodist/mo/doctor-cabinet?doctor_key={doctor_b}",
        headers=doctor_headers,
    )
    assert denied.status_code == 403
    assert client.get("/api/methodist/mo/cases", headers=doctor_headers).status_code == 403

    monkeypatch.setenv("MO_TRUST_DOCTOR_IDENTITY_HEADER", "true")
    monkeypatch.setenv("MO_DOCTOR_IDENTITY_SHARED_SECRET", "proxy-secret")
    spoofed = client.get(
        "/api/methodist/mo/doctor-cabinet",
        headers={
            **doctor_headers,
            "X-Trusted-Doctor-Key": doctor_a,
            "X-Mo-Identity-Secret": "wrong",
        },
    )
    assert spoofed.status_code == 403
    allowed = client.get(
        f"/api/methodist/mo/doctor-cabinet?doctor_key={doctor_b}",
        headers={
            **doctor_headers,
            "X-Trusted-Doctor-Key": doctor_a,
            "X-Mo-Identity-Secret": "proxy-secret",
        },
    )
    assert allowed.status_code == 200
    payload = allowed.json()
    assert payload["doctor"]["doctor_key"] == doctor_a
    assert {item["mis_id"] for item in payload["cases"]} == {"case-1"}
    assert {item["mis_id"] for item in payload["findings"]} == {"case-1"}


def test_methodist_may_open_any_doctor_and_fio_is_visible(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    _, doctor_b = _seed(db)
    monkeypatch.setenv("METHODIST_TOKEN", "test-token")
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    response = TestClient(rag_server.app).get(
        f"/api/methodist/mo/doctor-cabinet?doctor_key={doctor_b}",
        headers={"X-Methodist-Token": "test-token", "X-Methodist-Role": "methodist"},
    )
    assert response.status_code == 200
    assert response.json()["doctor"]["doctor_fio"] == "Врач Б"


def test_dispute_persists_event_without_overwriting_crm_state(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    doctor_a, _ = _seed(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    result = mo_backend.create_dispute(
        actor=doctor_a,
        role="doctor",
        doctor_key=doctor_a,
        case_id="visit-1",
        finding_code="B_gap",
        reason="Не согласен с трактовкой источника",
    )
    assert result["status"] == "submitted"
    with sqlite3.connect(db) as conn:
        event = conn.execute(
            "SELECT event_type,actor,payload_json FROM crm_case_event WHERE event_id=?",
            (result["event_id"],),
        ).fetchone()
        state = conn.execute(
            "SELECT status,assignee,tags_json FROM crm_case_state WHERE case_id='visit-1'"
        ).fetchone()
        dispute_state = conn.execute(
            "SELECT status,reason,actor FROM crm_dispute_state WHERE dispute_id=?",
            (result["dispute_id"],),
        ).fetchone()
    assert event[0] == "doctor_dispute"
    assert json.loads(event[2])["reason"] == "Не согласен с трактовкой источника"
    assert state == ("in_review", "Методист", '["важно"]')
    assert dispute_state == (
        "submitted",
        "Не согласен с трактовкой источника",
        doctor_a,
    )
    assert mo_backend.build_doctor_cabinet(
        doctor_key=doctor_a, actor="Методист", role="methodist"
    )["dispute_stats"]["total"] == 1


def test_access_audit_is_admin_only_and_excludes_sensitive_data(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    doctor_a, _ = _seed(db)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    mo_backend.record_access(
        actor="Администратор",
        role="admin",
        action="doctor_personal_export",
        doctor_key=doctor_a,
        metadata={"token": "secret", "patient_text": "clinical", "job_id": "job-1"},
    )
    try:
        mo_backend.list_access_log(role="methodist")
    except PermissionError:
        pass
    else:
        raise AssertionError("Журнал доступа должен быть только для admin")
    item = mo_backend.list_access_log(role="admin")["items"][0]
    assert item["metadata"] == {"job_id": "job-1"}
    assert "secret" not in json.dumps(item, ensure_ascii=False)
    assert "clinical" not in json.dumps(item, ensure_ascii=False)
    export = mo_backend.create_doctor_export(
        doctor_key=doctor_a, actor="Администратор", role="admin"
    )
    assert mo_backend.get_export(actor="Администратор", job_id=export["job_id"]).is_file()
    actions = {
        row["action"] for row in mo_backend.list_access_log(role="admin")["items"]
    }
    assert {"doctor_cabinet_open", "doctor_personal_export"} <= actions


def test_access_log_http_requires_server_admin_secret(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    _seed(db)
    monkeypatch.setenv("METHODIST_TOKEN", "test-token")
    monkeypatch.setenv("MO_ADMIN_TOKEN", "admin-secret")
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    client = TestClient(rag_server.app)
    base_headers = {
        "X-Methodist-Token": "test-token",
        "X-Methodist-Role": "admin",
    }
    assert client.get("/api/methodist/mo/access-log", headers=base_headers).status_code == 403
    assert client.get(
        "/api/methodist/mo/access-log",
        headers={**base_headers, "X-Methodist-Admin-Token": "wrong"},
    ).status_code == 403
    assert client.get(
        "/api/methodist/mo/access-log",
        headers={**base_headers, "X-Methodist-Admin-Token": "admin-secret"},
    ).status_code == 200


def test_template_detector_threshold_privacy_advisory_and_no_score_penalty(
    monkeypatch, tmp_path: Path
) -> None:
    common = (
        "Пациент жалуется на боль назначено обследование контроль через семь дней "
        "состояние стабильное рекомендации разъяснены динамическое наблюдение врача "
        "повторный осмотр после получения результатов лабораторных исследований "
        "лечение согласовано противопоказания и аллергические реакции отрицает"
    )
    rows = [
        {
            "id": "101",
            "visit_id": "201",
            "visit_date": "2026-07-29",
            "patient_id": "p1",
            "doctor_fio": "Врач А",
            "doctor_specialization": "Терапия",
            "document_kind": "clinical_visit",
            "complaints": common,
            "anamnesis_doctor": common,
        },
        {
            "id": "102",
            "visit_id": "202",
            "visit_date": "2026-07-29",
            "patient_id": "p2",
            "doctor_fio": "Врач А",
            "doctor_specialization": "Терапия",
            "document_kind": "clinical_visit",
            "complaints": common,
            "anamnesis_doctor": common,
        },
        {
            "id": "103",
            "visit_id": "203",
            "visit_date": "2026-07-29",
            "patient_id": "p3",
            "doctor_fio": "Врач А",
            "doctor_specialization": "Терапия",
            "document_kind": "clinical_visit",
            "complaints": "Совершенно другой короткий клинический текст без совпадений",
        },
    ]
    pairs = detect_template_copies(rows, threshold=0.85)
    assert {(item["case_id_a"], item["case_id_b"]) for item in pairs} == {("101", "102")}
    assert common not in json.dumps(pairs, ensure_ascii=False)

    db = tmp_path / "warehouse.sqlite"
    cases = [
        {"mis_id": row["id"], "visit_id": row["visit_id"], "overall_pct": 88.0, "doctor_fio": "Врач А", "deep": {}}
        for row in rows
    ]
    upsert_warehouse(
        db,
        rows,
        cases,
        {"date": "2026-07-29", "revision": 1, "quality": {"passed": True}},
    )
    with sqlite3.connect(db) as conn:
        finding = conn.execute(
            """SELECT severity,evidence,source_ref FROM fact_mo_finding
               WHERE mis_id='101' AND finding_code='E_template_copy'"""
        ).fetchone()
        score = conn.execute("SELECT overall_pct FROM fact_mo_case WHERE mis_id='101'").fetchone()[0]
        pair_json = conn.execute("SELECT provenance_json FROM fact_mo_template_pair").fetchone()[0]
    assert finding[0] == "P2"
    assert finding[1] == ""
    assert finding[2].startswith("template_pair:")
    assert score == 88.0
    assert common not in pair_json
