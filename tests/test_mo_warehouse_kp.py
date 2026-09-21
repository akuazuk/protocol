from __future__ import annotations

import sqlite3
from pathlib import Path

from clinical_knowledge.mo_daily import (
    initialize_warehouse,
    protocol_identity_from_suggest,
    resolve_case_protocol_suggest,
    upsert_warehouse,
    warehouse_protocol_suggest_live_enabled,
)
from tests.test_mo_zone_scores import _rich_clinical


def test_protocol_identity_reads_clinical_hit_not_top_level() -> None:
    protocol_id, version = protocol_identity_from_suggest(
        {
            "ok": True,
            "items": [
                {
                    "protocol_id": "tonsillitis_t1",
                    "title": "Острый тонзиллит",
                    "match_kind": "clinical",
                    "score": 80,
                    "document_date": "2026-03-01",
                }
            ],
        }
    )
    assert protocol_id == "tonsillitis_t1"
    assert version == "2026-03-01"


def test_protocol_identity_empty_without_clinical_hit() -> None:
    protocol_id, version = protocol_identity_from_suggest(
        {
            "items": [
                {
                    "protocol_id": "specialty_only",
                    "match_kind": "specialty",
                    "score": 90,
                }
            ]
        }
    )
    assert protocol_id == ""
    assert version == ""


def test_warehouse_persists_protocol_id_from_clinical_hit(tmp_path: Path) -> None:
    path = tmp_path / "wh.sqlite"
    initialize_warehouse(path)
    raw_rows = [
        {
            "id": "1",
            "visit_id": "v1",
            "visit_date": "2026-08-02",
            "document_kind": "clinical_visit",
            "doctor_fio": "Иванов И.И.",
            "doctor_specialization": "Терапия",
            "filial": "1",
            "patient_id": "p1",
            "visit_time": "10:15",
            **_rich_clinical(),
            "mkb_code_main": "J03.9",
        }
    ]
    cases = [
        {
            "mis_id": "1",
            "visit_id": "v1",
            "overall_pct": 80,
            "status": "good",
            "block_scores": {"exams": 70, "treatment": 65},
            "protocol_suggest": {
                "items": [
                    {
                        "title": "Острый тонзиллит",
                        "match_kind": "clinical",
                        "score": 80,
                        "protocol_id": "t1",
                    }
                ]
            },
            "evaluation_v4": {"scorer_version": "test", "schema_version": "1", "findings": []},
            **_rich_clinical(),
        }
    ]
    report = {"date": "2026-08-02", "quality": {"passed": True}, "partial": False}
    upsert_warehouse(path, raw_rows, cases, report)
    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT protocol_id, zone2b_kp_status, protocol_applicability_status "
            "FROM fact_mo_case WHERE mis_id='1'"
        ).fetchone()
    assert row == ("t1", "matched", "applicable")


def test_warehouse_runs_live_suggest_when_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MO_WAREHOUSE_PROTOCOL_SUGGEST", "1")
    called = {}

    def fake_suggest(**kwargs):
        called["clinical"] = kwargs.get("clinical") or {}
        return {
            "ok": True,
            "items": [
                {
                    "title": "Острый тонзиллит",
                    "match_kind": "clinical",
                    "score": 77,
                    "protocol_id": "live_t1",
                }
            ],
        }

    monkeypatch.setattr(
        "clinical_knowledge.case_protocol_suggest.suggest_protocols_for_mo_case",
        fake_suggest,
    )
    path = tmp_path / "wh.sqlite"
    initialize_warehouse(path)
    raw_rows = [
        {
            "id": "1",
            "visit_id": "v1",
            "visit_date": "2026-08-02",
            "document_kind": "clinical_visit",
            "doctor_fio": "Иванов И.И.",
            "doctor_specialization": "Терапия",
            "filial": "1",
            "patient_id": "p1",
            "visit_time": "10:15",
            **_rich_clinical(),
            "mkb_code_main": "J03.9",
        }
    ]
    cases = [
        {
            "mis_id": "1",
            "visit_id": "v1",
            "overall_pct": 80,
            "status": "good",
            "block_scores": {"exams": 70, "treatment": 65},
            "evaluation_v4": {"scorer_version": "test", "schema_version": "1", "findings": []},
            **_rich_clinical(),
        }
    ]
    report = {"date": "2026-08-02", "quality": {"passed": True}, "partial": False}
    upsert_warehouse(path, raw_rows, cases, report)
    assert called.get("clinical")
    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT protocol_id, zone2b_kp_status, protocol_applicability_status "
            "FROM fact_mo_case WHERE mis_id='1'"
        ).fetchone()
    assert row == ("live_t1", "matched", "applicable")


def test_pytest_does_not_call_live_suggest_by_default() -> None:
    assert warehouse_protocol_suggest_live_enabled() is False
    assert resolve_case_protocol_suggest({"mis_id": "1"}) is None
