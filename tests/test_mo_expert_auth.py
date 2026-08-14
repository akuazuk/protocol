"""Кабинет врача-эксперта: auth, capabilities, reports min_date, review pack source."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from clinical_knowledge import mo_backend, mo_expert_auth, mo_review_pack
from clinical_knowledge.mo_daily import CRM_TABLES, initialize_warehouse


def _seed_case(db: Path, *, visit_id: str = "3646270", day: str = "2026-08-04") -> None:
    initialize_warehouse(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO fact_mo_case(
                 mis_id, visit_id, visit_date, document_kind, overall_pct, status,
                 scorer_version, score_schema_version, doctor_key, specialty, filial,
                 diagnosis_code, icd_chapter, content_hash, updated_at
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "898517",
                visit_id,
                day,
                "clinical_visit",
                62.0,
                "review",
                "v3",
                "1",
                "doc-1",
                "Терапия",
                "Филиал Центр",
                "J06.9",
                "X",
                "hash",
                "2026-08-04T12:00:00Z",
            ),
        )
        conn.commit()


def test_crm_tables_include_expert() -> None:
    assert "crm_expert_user" in CRM_TABLES
    assert "crm_expert_session" in CRM_TABLES


def test_expert_login_logout_and_path_allowlist(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    initialize_warehouse(db)

    created = mo_expert_auth.upsert_expert_user(
        login="expert",
        password="test-pass-123",
        display_name="Тест Эксперт",
    )
    assert created["ok"] is True

    session = mo_expert_auth.login_expert(login="expert", password="test-pass-123")
    assert session["ok"] is True
    assert session["role"] == "expert"
    assert session["session_token"]
    assert session["reports_min_date"] == "2026-08-01"

    headers = {"X-Expert-Session": session["session_token"]}
    resolved = mo_expert_auth.resolve_expert_session(headers)
    assert resolved is not None
    assert resolved["login"] == "expert"
    assert resolved["actor"] == "expert:expert"

    assert mo_expert_auth.expert_path_allowed("/api/methodist/mo/daily-report")
    assert mo_expert_auth.expert_path_allowed("/api/methodist/mo/cases/1/review-pack")
    assert not mo_expert_auth.expert_path_allowed("/api/methodist/mo/month-report")
    assert not mo_expert_auth.expert_path_allowed("/api/methodist/mo/heatmap")

    mo_expert_auth.logout_expert(session["session_token"])
    assert mo_expert_auth.resolve_expert_session(headers) is None


def test_expert_capabilities_and_reports_min_date(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MO_ANALYTICS_DB", str(tmp_path / "mo.sqlite"))
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_EXPERT_REPORTS_MIN_DATE", "2026-08-01")
    initialize_warehouse(tmp_path / "mo.sqlite")

    caps = mo_backend.build_mo_capabilities("expert")
    assert caps["role"] == "expert"
    assert caps["reports_min_date"] == "2026-08-01"
    assert caps["pages"]["yesterday"] is True
    assert caps["pages"]["reports"] is True
    assert caps["pages"]["overview"] is False
    assert caps["pages"]["queue"] is False
    assert caps["pages"]["kp_sync"] is False
    assert caps["pages"]["rceth_sync"] is False
    assert caps["actions"]["review_pack"] is True
    assert caps["actions"]["bulk_action"] is False

    reports_root = tmp_path / "medical_exams" / "reports" / "2026" / "07" / "2026-07-15"
    reports_root.mkdir(parents=True)
    (reports_root / "report.json").write_text(
        '{"date":"2026-07-15","summary":{"source_rows":10,"scored":8},"quality_status":"ok"}',
        encoding="utf-8",
    )
    aug = tmp_path / "medical_exams" / "reports" / "2026" / "08" / "2026-08-02"
    aug.mkdir(parents=True)
    (aug / "report.json").write_text(
        '{"date":"2026-08-02","summary":{"source_rows":12,"scored":9},"quality_status":"ok"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mo_backend,
        "_medical_exam_roots",
        lambda: [tmp_path / "medical_exams"],
    )
    monkeypatch.setattr(mo_backend, "_warehouse_available", lambda: False)

    filtered = mo_backend.build_reports(min_date="2026-08-01")
    dates = [item["date"] for item in filtered["items"]]
    assert "2026-08-02" in dates
    assert "2026-07-15" not in dates
    assert filtered["min_date"] == "2026-08-01"


def test_expert_review_pack_forces_source(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    _seed_case(db)

    saved = mo_review_pack.save_review_pack(
        case_id="3646270",
        actor="expert:expert",
        role="expert",
        decision={
            "status": "confirmed_issue",
            "verdict_completeness": "agree",
            "verdict_diagnosis": "agree",
            "verdict_recommendations": "partial",
            "summary_ru": "Экспертный разбор",
            "source": "methodist",
            "training_use": False,
        },
    )
    assert saved["ok"] is True
    pack = mo_review_pack.get_review_pack(saved["pack_id"])["pack"]
    assert pack["decision"]["source"] == "expert"
    assert pack["decision"]["training_use"] is True
    assert pack["actor"].startswith("expert:")
