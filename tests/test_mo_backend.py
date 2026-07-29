from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

import rag_server
from clinical_knowledge import mo_backend
from clinical_knowledge.mis_kz_quality import classify_document_kind


def _record(case_id: str, *, patient_id: str = "secret", specialty: str = "Терапевт") -> dict:
    return {
        "case_id": case_id,
        "visit_id": case_id,
        "patient_id": patient_id,
        "date": "2026-07-27",
        "doctor_fio": "Врач Тестовый",
        "specialization": specialty,
        "filial": "Филиал 1",
        "kz_kind": "kz",
        "document_kind": "medical_exam",
        "document_kind_label": "Медицинский осмотр",
        "mkb_code_main": "Z00.0",
        "icd_chapter": "XXI",
        "icd_chapter_label": "Z00-Z99 Факторы здоровья",
        "diagnosis_short": "Профилактический осмотр",
        "overall_pct": 80.0,
        "status": "good",
        "score_band": "75-90",
        "axis_documentation": 80.0,
        "axis_concordance": 80.0,
        "axis_safety": 80.0,
        "axis_regulatory": 80.0,
        "p0": 0,
        "p1": 0,
        "p2": 0,
        "p3": 0,
        "finding_axes": [],
        "mkb_code_agreement": "match",
        "age_group": "взрослые (18-64)",
        "parse_ok": "1",
        "date_mismatch": "0",
    }


def test_document_kind_is_additive_and_uses_taxonomy_rules() -> None:
    row = {
        "kz_kind": "kz",
        "pay_type_label": "Справки и профосмотры",
        "service_names": "Медицинский осмотр",
    }
    assert classify_document_kind(row, {"fields_present": {"objective_status": True}}) == "medical_exam"
    assert row["kz_kind"] == "kz"
    assert classify_document_kind({}, {"text_len": 0, "fields_present": {}}) == "empty"
    assert classify_document_kind({"service_names": "УЗИ органов"}, {}) == "diagnostic"


def test_cases_hide_patient_id_and_suppress_small_groups(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MO_ANALYTICS_DB", str(tmp_path / "mo.sqlite"))
    monkeypatch.setattr(mo_backend, "_records", lambda params: [_record("1"), _record("2")])
    result = mo_backend.build_cases({"page": 1, "page_size": 50})
    assert result["total"] == 2
    assert "patient_id" not in result["rows"][0]
    assert result["rows"][0]["kz_kind"] == "kz"
    assert result["rows"][0]["document_kind"] == "medical_exam"
    assert result["aggregate"]["by_specialty"][0]["suppressed"] is True
    assert result["aggregate"]["by_specialty"][0]["n"] is None
    overview = mo_backend.build_overview({})
    assert overview["kpi"]["suppressed"] is True
    assert overview["kpi"]["avg_score"] is None


def test_crm_state_events_and_saved_views_are_persistent(monkeypatch, tmp_path) -> None:
    db = tmp_path / "mo.sqlite"
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    result = mo_backend.apply_bulk_action(
        actor="ИП",
        role="methodist",
        payload={
            "case_ids": ["case-1", "case-2"],
            "changes": {
                "status": "in_review",
                "assignee": "ИП",
                "tags": ["P1"],
                "finding_decisions": {"B_exams_gap": "confirmed"},
            },
            "comment": "Проверить",
        },
    )
    assert result["updated"] == 2
    saved = mo_backend.save_view(
        actor="ИП",
        payload={"name": "Моя очередь", "scope": "private", "filters": {"crm_statuses": ["in_review"]}},
    )
    assert mo_backend.list_views("ИП")["items"][0]["filters"]["crm_statuses"] == ["in_review"]
    assert mo_backend.delete_view(actor="ИП", view_id=saved["view_id"])["ok"] is True
    assert mo_backend.list_views("ИП")["items"] == []
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM crm_case_state").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM crm_case_event").fetchone()[0] == 2
        assert conn.execute("SELECT status FROM crm_case_state WHERE case_id='case-1'").fetchone()[0] == "in_review"
        assert "B_exams_gap" in conn.execute(
            "SELECT finding_decisions_json FROM crm_case_state WHERE case_id='case-1'"
        ).fetchone()[0]


def test_daily_report_prefers_pipeline_artifact(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    report_dir = tmp_path / "reports" / "2026" / "07" / "27"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "date": "2026-07-27",
                "revision": 2,
                "generated_at": "2026-07-28T05:00:00Z",
                "quality": {"passed": True},
                "summary": {
                    "source_rows": 580,
                    "eligible_rows": 504,
                    "scored": 500,
                    "avg_score": 78.4,
                    "needs_attention": 90,
                    "critical": 12,
                },
                "axes": {"documentation": 81.0},
                "action_queue": [{"mis_id": "1", "priority": "P1"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = mo_backend.build_daily_report("2026-07-27")
    assert result["revision"] == 2
    assert result["executive_summary"]["kpi"]["n"] == 580
    assert result["quality_status"] == "ok"
    assert result["action_queue"][0]["priority"] == "P1"


def test_daily_pipeline_cases_extend_monthly_analytics(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    secure = tmp_path / "secure_cases" / "2026" / "07"
    secure.mkdir(parents=True)
    (secure / "mo_2026-07-27.csv").write_text(
        "visit_id,visit_date,doctor_fio,doctor_specialization,filial,document_kind,parse_ok\n"
        "77,2026-07-27,Врач Новый,Невролог,Филиал 2,medical_exam,1\n",
        encoding="utf-8",
    )
    (secure / "kz_l1_2026-07-27_cases.jsonl").write_text(
        json.dumps(
            {
                "visit_id": "77",
                "overall_pct": 64.0,
                "deep": {
                    "overall_pct": 64.0,
                    "status": "review",
                    "axes": {"documentation": 70},
                    "findings": [{"code": "B_exam_gap", "severity": "P1"}],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    mo_backend._pipeline_records_for_month.cache_clear()
    rows = mo_backend._pipeline_records_for_month("2026-07")
    assert len(rows) == 1
    assert rows[0]["case_id"] == "77"
    assert rows[0]["_source"] == "daily_pipeline"
    assert rows[0]["document_kind"] == "medical_exam"
    monkeypatch.setenv("MO_ANALYTICS_DB", str(tmp_path / "crm.sqlite"))
    detail = mo_backend.build_case_detail("77", month="2026-07")
    assert detail["source"] == "daily_pipeline"
    assert detail["findings"][0]["code"] == "B_exam_gap"
    assert "_source" not in detail["record"]


def test_freshness_reports_lag_and_empty_reason(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    report_dir = tmp_path / "reports" / "2026" / "07" / "27"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps({"date": "2026-07-27", "generated_at": "2026-07-28T05:00:00Z", "revision": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "pipeline.json").write_text(
        json.dumps({"dates": {"2026-07-27": {"status": "success", "heartbeat": "2026-07-28T05:00:00Z"}}, "runs": [1]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mo_backend, "_records", lambda params: [])
    payload = mo_backend.build_freshness({})
    assert payload["ok"] is True
    assert payload["latest_report"]["date"] == "2026-07-27"
    assert payload["state"]["status"] == "present"
    assert payload["empty_state"]["reason_code"] == "no_source_data"


def test_queue_only_keeps_risky_cases(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MO_ANALYTICS_DB", str(tmp_path / "mo.sqlite"))
    good = _record("1")
    low = {**_record("2"), "overall_pct": 60.0}
    risky = {**_record("3"), "p1": 1}
    monkeypatch.setattr(mo_backend, "_records", lambda params: [good, low, risky])
    result = mo_backend.build_cases({"queue_only": True})
    assert {row["case_id"] for row in result["rows"]} == {"2", "3"}


def test_export_job_is_private_and_downloadable(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MO_ANALYTICS_DB", str(tmp_path / "mo.sqlite"))
    monkeypatch.setattr(mo_backend, "_records", lambda params: [_record("1")])
    job = mo_backend.create_export(actor="ИП", payload={"kind": "cases", "filters": {}})
    path = mo_backend.get_export(actor="ИП", job_id=job["job_id"])
    assert path.is_file()
    assert job["download_url"].endswith(job["job_id"])
    try:
        mo_backend.get_export(actor="Другой", job_id=job["job_id"])
    except PermissionError:
        pass
    else:
        raise AssertionError("чужая выгрузка должна быть недоступна")


def test_mo_api_uses_methodist_auth_and_no_store(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("METHODIST_TOKEN", "mo-test-token")
    monkeypatch.setenv("MO_ANALYTICS_DB", str(tmp_path / "mo.sqlite"))
    monkeypatch.setattr(
        mo_backend,
        "build_case_detail",
        lambda case_id, month=None: {"ok": True, "case_id": case_id, "record": {"kz_kind": "kz", "document_kind": "consultation"}},
    )
    monkeypatch.setattr(
        mo_backend,
        "build_daily_report",
        lambda report_date: {"ok": True, "date": report_date},
    )
    client = TestClient(rag_server.app)
    assert client.get("/api/methodist/mo/cases/123").status_code == 403
    response = client.get(
        "/api/methodist/mo/cases/123",
        headers={"X-Methodist-Token": "mo-test-token"},
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json()["record"]["kz_kind"] == "kz"
    daily = client.get(
        "/api/methodist/mo/daily-report?date=2026-07-27",
        headers={"X-Methodist-Token": "mo-test-token"},
    )
    assert daily.status_code == 200
    assert daily.headers["cache-control"] == "private, no-store"
    monkeypatch.setattr(mo_backend, "_records", lambda params: [_record("1")])
    cases = client.get(
        "/api/methodist/mo/cases?queue_only=true",
        headers={"X-Methodist-Token": "mo-test-token"},
    )
    assert cases.status_code == 200
    assert cases.json()["applied_filters"] == {"queue_only": True, "page": 1, "page_size": 50, "sort_by": "date", "sort_dir": "desc"}
    freshness = client.get(
        "/api/methodist/mo/freshness",
        headers={"X-Methodist-Token": "mo-test-token"},
    )
    assert freshness.status_code == 200
    assert freshness.json()["ok"] is True


def test_mo_mutations_reject_viewer_role(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("METHODIST_TOKEN", "mo-test-token")
    monkeypatch.setenv("MO_ANALYTICS_DB", str(tmp_path / "mo.sqlite"))
    client = TestClient(rag_server.app)
    response = client.post(
        "/api/methodist/mo/cases/bulk-action",
        headers={"X-Methodist-Token": "mo-test-token", "X-Methodist-Role": "viewer"},
        json={"case_ids": ["1"], "changes": {"status": "in_review"}},
    )
    assert response.status_code == 403


def test_legacy_compatibility_metadata_contract() -> None:
    meta = mo_backend.compatibility_metadata()
    assert meta["deprecated"] is True
    assert meta["replacement"] == "/api/methodist/mo"
    assert {"kz_kind", "evaluation_v3"} <= set(meta["legacy_fields_preserved"])
