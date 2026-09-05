"""Review pack методиста: схема, save/list/revise, patient_id enrichment."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from clinical_knowledge import mo_backend, mo_review_pack
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
        conn.execute(
            """INSERT OR REPLACE INTO dim_doctor(doctor_key, doctor_fio, specialty, filial)
               VALUES (?,?,?,?)""",
            ("doc-1", "Иванов И.И.", "Терапия", "Филиал Центр"),
        )
        conn.commit()


def test_crm_tables_include_review_pack() -> None:
    assert "crm_review_pack" in CRM_TABLES


def test_save_list_and_revise_review_pack(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MO_BACKEND_SOURCE", "warehouse")
    _seed_case(db)

    secure = tmp_path / "secure_cases" / "2026" / "08"
    secure.mkdir(parents=True)
    with (secure / "mo_2026-08-04.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "visit_id",
                "id",
                "mis_id",
                "patient_id",
                "complaints",
                "clinical_diagnosis",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "visit_id": "3646270",
                "id": "898517",
                "mis_id": "898517",
                "patient_id": "PAT-77",
                "complaints": "кашель",
                "clinical_diagnosis": "ОРВИ",
            }
        )

    saved = mo_review_pack.save_review_pack(
        case_id="3646270",
        actor="Методист",
        role="methodist",
        decision={
            "status": "confirmed_issue",
            "verdict_completeness": "partial",
            "verdict_diagnosis": "agree",
            "verdict_recommendations": "disagree",
            "summary_ru": "Нужно уточнить план лечения",
            "training_use": True,
            "corrected_scores": {"recommendations": 30},
        },
    )
    assert saved["ok"] is True
    assert saved["patient_id"] == "PAT-77"
    assert saved["pack_id"]

    listed = mo_review_pack.list_review_packs("3646270")
    assert listed["ok"] is True
    assert len(listed["items"]) == 1
    assert listed["items"][0]["decision_summary"]["verdict_diagnosis"] == "agree"

    full = mo_review_pack.get_review_pack(saved["pack_id"])
    assert full["ok"] is True
    assert full["pack"]["clinical"]["complaints"] == "кашель"
    assert full["pack"]["decision"]["corrected_scores"]["recommendations"] == 30

    revised = mo_review_pack.revise_review_pack(
        pack_id=saved["pack_id"],
        actor="Методист",
        role="methodist",
        decision={"verdict_recommendations": "partial", "summary_ru": "План ок после правки"},
    )
    assert revised["ok"] is True
    assert revised["supersedes_pack_id"] == saved["pack_id"]
    listed2 = mo_review_pack.list_review_packs("3646270")
    assert len(listed2["items"]) == 2

    with sqlite3.connect(db) as conn:
        status = conn.execute(
            "SELECT status FROM crm_case_state WHERE case_id='3646270'"
        ).fetchone()[0]
        assert status == "confirmed_issue"
        event = conn.execute(
            "SELECT event_type FROM crm_case_event WHERE case_id='3646270' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()[0]
        assert event == "review_pack_saved"


def test_build_cases_can_include_patient_id(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MO_ANALYTICS_DB", str(tmp_path / "mo.sqlite"))
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(
        mo_backend,
        "_records",
        lambda params: [
            {
                "case_id": "1",
                "visit_id": "1",
                "patient_id": "secret",
                "date": "2026-08-04",
                "doctor_fio": "Врач",
                "specialization": "Терапия",
                "filial": "А",
                "document_kind": "clinical_visit",
                "document_kind_label": "КЗ",
                "overall_pct": 70.0,
                "status": "review",
                "p0": 0,
                "p1": 1,
            }
        ],
    )
    hidden = mo_backend.build_cases({"page": 1, "page_size": 10})
    assert "patient_id" not in hidden["rows"][0]
    shown = mo_backend.build_cases({"page": 1, "page_size": 10, "include_patient_id": True})
    assert shown["rows"][0]["patient_id"] == "secret"
    assert shown["rows"][0]["visit_id"] == "1"


def test_patient_id_map_for_day(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    day_dir = tmp_path / "secure_cases" / "2026" / "08"
    day_dir.mkdir(parents=True)
    with (day_dir / "mo_2026-08-04.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["visit_id", "mis_id", "patient_id", "doctor_id", "doctor_fio", "filial"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "visit_id": "10",
                "mis_id": "20",
                "patient_id": "P1",
                "doctor_id": "991",
                "doctor_fio": "",
                "filial": "Центр",
            }
        )
    mapping = mo_review_pack.patient_id_map_for_day("2026-08-04")
    assert mapping["10"] == "P1"
    assert mapping["20"] == "P1"
    identity = mo_review_pack.visit_identity_map_for_day("2026-08-04")
    assert identity["10"]["doctor_id"] == "991"
    assert identity["10"]["filial"] == "Центр"
    rows = [{"case_id": "10", "visit_id": "10", "date": "2026-08-04", "doctor": "Врач не указан"}]
    mo_review_pack.enrich_rows_with_patient_id(rows)
    assert rows[0]["doctor_id"] == "991"
    assert rows[0]["filial"] == "Центр"


def test_save_review_pack_requires_methodist_role(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "mo.sqlite"
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    _seed_case(db)
    with pytest.raises(PermissionError, match="methodist"):
        mo_review_pack.save_review_pack(
            case_id="3646270",
            actor="viewer",
            role="viewer",
            decision={"status": "in_review"},
        )
