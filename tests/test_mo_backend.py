from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

import rag_server
from clinical_knowledge import mo_backend
from clinical_knowledge.mo_daily import doctor_key_for, initialize_warehouse
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
        "document_kind": "clinical_visit",
        "document_kind_label": "Клинический приём",
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
    assert (
        classify_document_kind(
            {
                "kz_kind": "kz",
                "service_names": "Консультация врача-терапевта",
                "complaints": "кашель",
                "objective_status": "дыхание везикулярное",
                "clinical_diagnosis": "J06",
            },
            {},
        )
        == "clinical_visit"
    )


def test_cases_hide_patient_id_and_suppress_small_groups(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MO_ANALYTICS_DB", str(tmp_path / "mo.sqlite"))
    monkeypatch.setattr(mo_backend, "_records", lambda params: [_record("1"), _record("2")])
    result = mo_backend.build_cases({"page": 1, "page_size": 50})
    assert result["total"] == 2
    assert "patient_id" not in result["rows"][0]
    assert result["rows"][0]["kz_kind"] == "kz"
    assert result["rows"][0]["document_kind"] == "clinical_visit"
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


def test_case_detail_falls_back_to_warehouse_and_sanitizes_hash_diagnosis(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    initialize_warehouse(db)
    doctor = doctor_key_for("Тест Врач")
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO dim_doctor(doctor_key,doctor_fio,specialty,filial) VALUES(?,?,?,?)",
            (doctor, "Тест Врач", "Терапия", "Филиал"),
        )
        conn.execute(
            """INSERT INTO fact_mo_case
               (mis_id,visit_id,visit_date,document_kind,overall_pct,status,doctor_key,specialty,filial,
                diagnosis_code,icd_chapter,content_hash,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "hx-9001",
                "hx-91001",
                "2026-08-02",
                "clinical_visit",
                77.0,
                "review",
                doctor,
                "Терапия",
                "Филиал",
                "27b2db9e5a9f66a60ae6b378870abdb74ecaf847ed63d5fd17ed4d712b15b2a5",
                "N/A",
                "27b2db9e5a9f66a60ae6b378870abdb74ecaf847ed63d5fd17ed4d712b15b2a5",
                "2026-08-03T00:00:00Z",
            ),
        )
        conn.executemany(
            "INSERT INTO fact_mo_score_axis(mis_id,axis,score) VALUES(?,?,?)",
            [
                ("hx-9001", "documentation", 78.0),
                ("hx-9001", "clinical_concordance", 75.0),
                ("hx-9001", "safety", 80.0),
            ],
        )
        conn.commit()
    mo_backend._pipeline_records_for_month.cache_clear()
    detail = mo_backend.build_case_detail("hx-91001")
    assert detail["ok"] is True
    assert detail["source"] == "warehouse"
    assert detail["record"]["diagnosis_short"] == "Не указан"
    assert detail["record"]["diagnosis_code"] == ""
    assert isinstance(detail.get("coverage_pct"), float)
    assert isinstance(detail.get("confidence_pct"), float)
    assert detail["coverage_pct"] >= 70.0


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
        lambda case_id, month=None: {"ok": True, "case_id": case_id, "record": {"kz_kind": "kz", "document_kind": "clinical_visit"}},
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
    applied = cases.json()["applied_filters"]
    assert applied["queue_only"] is True
        assert applied["document_kinds"] == "clinical_visit|consultation"
    assert applied.get("score_eligible_only") in {"1", 1, True}
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


def _seed_analytics_warehouse(path: Path) -> None:
    initialize_warehouse(path)
    with sqlite3.connect(path) as conn:
        doctors = [
            ("Целевой врач", "Терапия", 60.0, 5),
            ("Коллега", "Терапия", 60.0, 5),
            ("Врач другой специальности", "Хирургия", 100.0, 10),
        ]
        mis_id = 1
        for fio, specialty, score, count in doctors:
            key = doctor_key_for(fio)
            conn.execute("INSERT INTO dim_doctor VALUES (?, ?, ?, ?)", (key, fio, specialty, "Филиал"))
            for _ in range(count):
                conn.execute(
                    """INSERT INTO fact_mo_case
                       (mis_id,visit_id,visit_date,document_kind,overall_pct,status,
                        doctor_key,specialty,filial,content_hash,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        str(mis_id),
                        str(mis_id),
                        "2026-07-15",
                        "clinical_visit",
                        score,
                        "good",
                        key,
                        specialty,
                        "Филиал",
                        f"hash-{mis_id}",
                        "2026-07-16T00:00:00Z",
                    ),
                )
                mis_id += 1
        conn.execute(
            """INSERT INTO fact_mo_case
               (mis_id,visit_id,visit_date,document_kind,overall_pct,status,
                doctor_key,specialty,filial,content_hash,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(mis_id),
                str(mis_id),
                "2026-07-15",
                "diagnostic",
                100.0,
                "good",
                "",
                "Диагностика",
                "Филиал",
                f"hash-{mis_id}",
                "2026-07-16T00:00:00Z",
            ),
        )
        for day, avg in (("2026-07-14", 70.0), ("2026-07-15", 80.0)):
            conn.execute(
                """INSERT INTO fact_mo_daily
                   (visit_date,source_rows,scored_rows,avg_score,revision,quality_status,updated_at,
                    eligible_rows,partial,coverage_pct,critical)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (day, 10, 10, avg, 1, "passed", "2026-07-16T00:00:00Z", 10, 0, 100.0, 0),
            )


def test_new_mo_endpoints_require_auth_and_use_seeded_warehouse(monkeypatch, tmp_path: Path) -> None:
    warehouse = tmp_path / "mo.sqlite"
    _seed_analytics_warehouse(warehouse)
    monkeypatch.setenv("METHODIST_TOKEN", "mo-test-token")
    monkeypatch.setenv("MO_ANALYTICS_DB", str(warehouse))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    client = TestClient(rag_server.app)
    headers = {"X-Methodist-Token": "mo-test-token"}
    urls = [
        "/api/methodist/mo/summary?period=custom&date_from=2026-07-15&date_to=2026-07-15&compare=previous",
        "/api/methodist/mo/timeseries?period=custom&date_from=2026-07-14&date_to=2026-07-15",
        "/api/methodist/mo/breakdown?period=custom&date_from=2026-07-15&date_to=2026-07-15&dimension=doctor",
        "/api/methodist/mo/heatmap?period=custom&date_from=2026-07-15&date_to=2026-07-15",
        "/api/methodist/mo/findings?period=custom&date_from=2026-07-15&date_to=2026-07-15",
        "/api/methodist/mo/meta",
    ]
    for url in urls:
        assert client.get(url).status_code == 403
        response = client.get(url, headers=headers)
        assert response.status_code == 200, (url, response.text)
        payload = response.json()
        assert payload["source"] == "warehouse"
        assert payload["schema_version"] == 1

    summary = client.get(urls[0], headers=headers).json()
    assert summary["periods"]["timezone"] == "Europe/Minsk"
    assert summary["deltas"]["source_records"] == 21
    assert summary["kpi"]["eligible"] == 20
    assert summary["kpi"]["evaluated"] == 20
    assert summary["kpi"]["coverage_pct"] == 100.0
    heatmap = client.get(urls[3], headers=headers).json()
    assert heatmap["status"] == "not_available"
    assert heatmap["cells"] == []


def test_doctor_expected_score_uses_specialty_not_clinic_mean(monkeypatch, tmp_path: Path) -> None:
    warehouse = tmp_path / "mo.sqlite"
    _seed_analytics_warehouse(warehouse)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(warehouse))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    result = mo_backend.build_breakdown(
        {
            "period": "custom",
            "date_from": "2026-07-15",
            "date_to": "2026-07-15",
            "dimension": "doctor",
            "sample_threshold": 5,
        }
    )
    target = next(item for item in result["items"] if item["label"] == "Целевой врач")
    assert target["avg_score"] == 60.0
    assert target["expected_score"] == 60.0
    assert target["delta"] == 0.0
    assert target["enough_data"] is True


def test_new_endpoints_reject_unknown_period_and_dimension(monkeypatch, tmp_path: Path) -> None:
    warehouse = tmp_path / "mo.sqlite"
    _seed_analytics_warehouse(warehouse)
    monkeypatch.setenv("METHODIST_TOKEN", "mo-test-token")
    monkeypatch.setenv("MO_ANALYTICS_DB", str(warehouse))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    client = TestClient(rag_server.app)
    headers = {"X-Methodist-Token": "mo-test-token"}
    bad_period = client.get("/api/methodist/mo/summary?period=quarter", headers=headers)
    assert bad_period.status_code == 422
    assert "Неизвестный period" in bad_period.text
    bad_dimension = client.get(
        "/api/methodist/mo/breakdown?dimension=patient",
        headers=headers,
    )
    assert bad_dimension.status_code == 422
    assert "Неизвестный dimension" in bad_dimension.text


def test_sql_summary_applies_shared_filters(monkeypatch, tmp_path: Path) -> None:
    warehouse = tmp_path / "mo.sqlite"
    _seed_analytics_warehouse(warehouse)
    monkeypatch.setenv("METHODIST_TOKEN", "mo-test-token")
    monkeypatch.setenv("MO_ANALYTICS_DB", str(warehouse))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    client = TestClient(rag_server.app)
    response = client.get(
        "/api/methodist/mo/summary"
        "?period=custom&date_from=2026-07-15&date_to=2026-07-15"
        "&specializations=Терапия",
        headers={"X-Methodist-Token": "mo-test-token"},
    )
    assert response.status_code == 200
    kpi = response.json()["kpi"]
    assert kpi["source_records"] == 10
    assert kpi["eligible"] == 10
    assert kpi["evaluated"] == 10
    assert kpi["avg_score"] == 60.0


def test_auto_source_falls_back_when_period_is_absent_from_warehouse(
    monkeypatch, tmp_path: Path
) -> None:
    warehouse = tmp_path / "mo.sqlite"
    initialize_warehouse(warehouse)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(warehouse))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "auto")
    assert (
        mo_backend._source_for_period(
            mo_backend.DateRange(date(2026, 7, 1), date(2026, 7, 31))
        )
        == "jsonl_fallback"
    )
