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


def _seed_yesterday(path: Path, report_root: Path) -> None:
    initialize_warehouse(path)
    chosen = "2026-07-29"
    previous = "2026-07-28"
    history = [
        "2026-06-03",
        "2026-06-10",
        "2026-06-17",
        "2026-06-24",
        "2026-07-01",
        "2026-07-08",
        "2026-07-15",
        "2026-07-22",
    ]
    low_key = doctor_key_for("Врач Ниже Ожидания")
    peer_key = doctor_key_for("Врач Коллега")
    now = "2026-07-30T05:00:00Z"
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT INTO dim_doctor VALUES (?, ?, ?, ?)",
            [
                (low_key, "Врач Ниже Ожидания", "Терапия", "Центр"),
                (peer_key, "Врач Коллега", "Терапия", "Север"),
            ],
        )
        conn.execute(
            "INSERT INTO dim_diagnosis(diagnosis_code,diagnosis_label,chapter) VALUES (?,?,?)",
            ("I10", "Артериальная гипертензия", "IX"),
        )
        for index in range(12):
            mis_id = f"day-{index}"
            doctor_key = low_key if index < 6 else peer_key
            branch = "Центр" if index < 6 else "Север"
            score = 50.0 if index < 6 else 90.0
            conn.execute(
                """INSERT INTO fact_mo_case
                   (mis_id,visit_id,visit_date,document_kind,overall_pct,status,
                    doctor_key,specialty,filial,diagnosis_code,icd_chapter,content_hash,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mis_id,
                    f"visit-{index}",
                    chosen,
                    "clinical_visit",
                    score,
                    "needs_review" if index < 6 else "good",
                    doctor_key,
                    "Терапия",
                    branch,
                    "I10",
                    "IX",
                    f"hash-{index}",
                    now,
                ),
            )
            if index < 6:
                conn.execute(
                    """INSERT INTO fact_mo_finding
                       (mis_id,finding_code,severity,passed,evidence,source_ref,title_ru)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        mis_id,
                        "C_red_flag",
                        "P1",
                        0,
                        "private evidence",
                        "private source",
                        "Красный флаг без маршрутизации",
                    ),
                )
            if index == 0:
                # №55 P1 не должен попадать в очередь разбора (только точные сигналы).
                conn.execute(
                    """INSERT INTO fact_mo_finding
                       (mis_id,finding_code,severity,passed,evidence,source_ref,title_ru)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        mis_id,
                        "D_reg55_gap",
                        "P1",
                        0,
                        "no dx slot",
                        "Пост. №55",
                        "Невыполненный критерий качества по постановлению МЗ № 55",
                    ),
                )
        for index in range(10):
            conn.execute(
                """INSERT INTO fact_mo_case
                   (mis_id,visit_id,visit_date,document_kind,overall_pct,status,
                    doctor_key,specialty,filial,diagnosis_code,icd_chapter,content_hash,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    f"prev-{index}",
                    f"prev-visit-{index}",
                    previous,
                    "clinical_visit",
                    75.0,
                    "good",
                    peer_key,
                    "Терапия",
                    "Центр" if index < 5 else "Север",
                    "I10",
                    "IX",
                    f"prev-hash-{index}",
                    now,
                ),
            )
        daily_rows = [(day, 100 + offset, 84.0 + offset / 10) for offset, day in enumerate(history)]
        daily_rows.extend([(previous, 10, 76.0), (chosen, 12, 70.0)])
        for day, source_rows, documentation in daily_rows:
            conn.execute(
                """INSERT INTO fact_mo_daily
                   (visit_date,source_rows,scored_rows,avg_score,revision,quality_status,updated_at,
                    eligible_rows,partial,coverage_pct,avg_documentation,
                    avg_clinical_concordance,avg_safety,avg_regulatory,needs_attention,critical)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    day,
                    source_rows,
                    source_rows,
                    70.0,
                    3,
                    "passed",
                    now,
                    source_rows,
                    0,
                    100.0,
                    documentation,
                    72.0,
                    96.0,
                    91.0,
                    6,
                    0,
                ),
            )
    report_dir = report_root / "reports" / "2026" / "07" / "29"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "date": chosen,
                "revision": 3,
                "generated_at": now,
                "partial": False,
                "quality": {
                    "passed": True,
                    "blocking": [],
                    "warnings": [
                        {
                            "code": "date_mismatch",
                            "severity": "warning",
                            "message": "Проверить расхождение дат",
                        }
                    ],
                    "metrics": {
                        "parse_ok_pct": 99.8,
                        "doctor_fio_filled_pct": 100.0,
                        "doctor_specialization_filled_pct": 100.0,
                        "filial_filled_pct": 100.0,
                        "mkb_code_main_filled_pct": 100.0,
                        "date_mismatch_pct": 0.2,
                    },
                },
                "completeness": {"coverage_pct": 100.0, "partial": False},
                "summary": {
                    "source_rows": 12,
                    "eligible_rows": 12,
                    "excluded_rows": 0,
                    "scored": 12,
                    "scoring_errors": 0,
                    "avg_score": 70.0,
                    "needs_attention": 6,
                    "critical": 0,
                },
                "axes": {
                    "documentation": 84.0,
                    "clinical_concordance": 72.0,
                    "safety": 96.0,
                    "regulatory": 91.0,
                },
                "action_queue": [{"visit_id": "visit-0", "reason": "legacy reason"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_yesterday_contract_combines_bounded_warehouse_and_report(monkeypatch, tmp_path: Path) -> None:
    warehouse = tmp_path / "warehouse.sqlite"
    _seed_yesterday(warehouse, tmp_path)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(warehouse))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))

    payload = mo_backend.build_daily_report("2026-07-29")

    assert payload["revision"] == 3
    assert payload["action_queue"][0]["reason"] == "legacy reason"
    assert payload["data_completeness"]["actual_rows"] == 12
    assert payload["data_completeness"]["expected_rows"]["samples"] == 8
    assert payload["funnel"]["source"] == 12
    assert payload["funnel"]["eligible"] == 12
    assert len(payload["indices"]["items"]) == 4
    assert all(item["available"] for item in payload["indices"]["items"])
    assert payload["top_findings"]["items"][0]["cases"] == 6
    label = payload["top_findings"]["items"][0]["label"]
    assert "C_red_flag" not in label
    assert "флаг" in label.lower() or "Красный" in label
    assert payload["top_findings"]["items"][0]["sample_cases"]
    assert payload["top_findings"]["day"] == "2026-07-29"
    assert len(payload["action_cases"]["items"]) == 6
    assert all(item["reason"] and item["case_id"] for item in payload["action_cases"]["items"])
    assert all("C_red_flag" not in item["finding_title"] for item in payload["action_cases"]["items"])
    assert all(item.get("finding_code") != "D_reg55_gap" for item in payload["action_cases"]["items"])
    assert all("P0" not in str(item.get("reason") or "") for item in payload["action_cases"]["items"])
    assert all(item.get("severity_label_ru") in {"Критично", "Важно"} for item in payload["action_cases"]["items"])
    # Seed: «Врач Ниже Ожидания» n=6 score=50 vs specialty peers 90 → delta < -10.
    # R² за день слабый - раньше жёсткий case_mix_reliable прятал весь блок.
    assert payload["doctor_outliers"]["available"] is True
    assert payload["doctor_outliers"]["items"]
    assert payload["doctor_outliers"]["items"][0]["label"] == "Врач Ниже Ожидания"
    assert float(payload["doctor_outliers"]["items"][0]["delta"]) < -10
    assert payload["doctor_outliers"].get("case_mix_soft") is True
    assert payload["flow_changes"]["dimensions"]["branch"]
    assert payload["source_quality"]["available"] is True
    drilldown = mo_backend.build_cases(
        {
            "date_from": "2026-07-29",
            "date_to": "2026-07-29",
            "finding_codes": "C_red_flag",
        }
    )
    assert drilldown["total"] == 6
    assert all("C_red_flag" in row["finding_codes"] for row in drilldown["rows"])


def test_yesterday_contract_suppresses_aggregates_and_does_not_leak_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    warehouse = tmp_path / "warehouse.sqlite"
    _seed_yesterday(warehouse, tmp_path)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(warehouse))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    payload = mo_backend.build_daily_report("2026-07-29")
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "private evidence" not in serialized
    assert "private source" not in serialized
    # patient_id в action_cases для методиста допустим; в warehouse seed значений нет
    for item in (payload.get("action_cases") or {}).get("items") or []:
        assert item.get("patient_id", "") == ""
        assert item.get("visit_id")
        assert item.get("visit_date")
    assert payload["suppression_n"] >= 5
    assert all(item["cases"] >= payload["suppression_n"] for item in payload["top_findings"]["items"])


def test_yesterday_unavailable_sections_are_explicit(monkeypatch, tmp_path: Path) -> None:
    warehouse = tmp_path / "empty.sqlite"
    initialize_warehouse(warehouse)
    monkeypatch.setenv("MO_ANALYTICS_DB", str(warehouse))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    payload = mo_backend.build_daily_report("2026-07-29")

    for key in (
        "data_completeness",
        "funnel",
        "indices",
        "top_findings",
        "action_cases",
        "doctor_outliers",
        "flow_changes",
        "source_quality",
    ):
        assert payload[key]["available"] is False
        assert payload[key]["reason"]


def test_yesterday_http_requires_auth_and_returns_private_contract(
    monkeypatch, tmp_path: Path
) -> None:
    warehouse = tmp_path / "warehouse.sqlite"
    _seed_yesterday(warehouse, tmp_path)
    monkeypatch.setenv("METHODIST_TOKEN", "yesterday-token")
    monkeypatch.setenv("MO_ANALYTICS_DB", str(warehouse))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    client = TestClient(rag_server.app)
    url = "/api/methodist/mo/daily-report?date=2026-07-29"

    assert client.get(url).status_code == 403
    response = client.get(url, headers={"X-Methodist-Token": "yesterday-token"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert len(response.json()["action_cases"]["items"]) >= 5


def test_yesterday_markup_rendering_and_minsk_date_contract() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    app = APP_PATH.read_text(encoding="utf-8")
    assert len(html.splitlines()) < 360
    for marker in (
        'id="yesterday-attention"',
        'id="yesterday-action-rows"',
        'id="yesterday-zone-trend"',
        'id="yesterday-completeness"',
        'id="yesterday-index-cards"',
        'id="yesterday-index-chart"',
        'id="yesterday-findings-chart"',
        'id="yesterday-doctor-chart"',
        'id="yesterday-flow-chart"',
        'id="yesterday-source-quality"',
    ):
        assert marker in html
    assert "взят в работу" in app
    assert "МО в PDF" in app
    assert "Europe/Minsk" in app
    assert "formatToParts" in app
    assert "Date.now() - 86400000" not in app
    assert "navigateYesterdayFinding" in app
    assert "data-open-case" in app or 'data-case="' in app
    assert "открыть список МО" in app
    for renderer in (
        "renderYesterdayIndices",
        "renderYesterdayFindings",
        "renderYesterdayDoctors",
        "renderYesterdayFlow",
        "renderAttentionStrip",
    ):
        assert renderer in app
    assert app.count("MO.moChart($(\"yesterday-") >= 3


def test_yesterday_javascript_syntax() -> None:
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
