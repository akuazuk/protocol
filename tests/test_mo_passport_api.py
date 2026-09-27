"""P3: API паспорта без patient_id и ФИО."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from clinical_knowledge.mo_daily import initialize_warehouse, patient_key_for
from clinical_knowledge.mo_patient_passport import (
    build_case_passport,
    build_passport_labs,
    build_patient_passport,
    clear_passport_cache,
    json_dumps_public,
    json_has_phi,
    passport_summary_for_case,
    rebuild_passports,
    resolve_patient_query,
)


def _insert_case(
    db: sqlite3.Connection,
    *,
    mis_id: str,
    visit_id: str,
    patient_key: str,
    visit_date: str,
    specialty: str,
    grade: str = "fair",
    code: str = "J03.9",
) -> None:
    db.execute(
        """
        INSERT INTO fact_mo_case(
          mis_id, visit_id, visit_date, document_kind, overall_pct,
          doctor_key, specialty, patient_key, diagnosis_code, diagnosis_text,
          overall_grade, content_hash, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            mis_id,
            visit_id,
            visit_date,
            "clinical_visit",
            70.0,
            "doc-a",
            specialty,
            patient_key,
            code,
            "Острый тонзиллит",
            grade,
            mis_id,
            "2026-09-27T00:00:00Z",
        ),
    )


def _setup(tmp_path: Path) -> tuple[Path, Path, str]:
    warehouse = tmp_path / "mo_analytics.sqlite"
    lab = tmp_path / "mo_lab.sqlite"
    initialize_warehouse(warehouse)
    pk = patient_key_for("p-api-1")
    with sqlite3.connect(warehouse) as db:
        _insert_case(
            db,
            mis_id="m-now",
            visit_id="v-2026",
            patient_key=pk,
            visit_date="2026-09-02",
            specialty="Терапевт",
            grade="fair",
        )
        _insert_case(
            db,
            mis_id="m-old",
            visit_id="v-2025",
            patient_key=pk,
            visit_date="2025-03-11",
            specialty="ЛОР",
            grade="poor",
            code="J32.0",
        )
        db.commit()
    with sqlite3.connect(lab) as db:
        db.execute(
            """
            CREATE TABLE fact_mo_lab (
              patient_key TEXT, test_date TEXT, test_id INTEGER,
              type_id INTEGER, type_name TEXT, indicator_id INTEGER,
              indicator_name TEXT, value TEXT, unit TEXT
            )
            """
        )
        db.execute(
            "INSERT INTO fact_mo_lab VALUES (?,?,?,?,?,?,?,?,?)",
            (pk, "2026-01-12", 1, 10, "ОАК", 1, "Hb", "130", "г/л"),
        )
        db.commit()
    rebuild_passports(warehouse, lab_path=lab)
    clear_passport_cache()
    return warehouse, lab, pk


def test_patient_passport_has_tiles_and_no_phi(tmp_path: Path) -> None:
    warehouse, lab, pk = _setup(tmp_path)
    payload = build_patient_passport(pk, warehouse=warehouse, lab_path=lab)
    assert payload["ok"] is True
    assert payload["patient_key"] == pk
    assert payload["coverage"]["n_visits"] == 2
    assert payload["coverage"]["n_specialties"] == 2
    assert len(payload["visits"]) == 2
    assert payload["visits"][0]["visit_id"] == "v-2026"
    assert "visits" in payload
    dumped = json_dumps_public(payload)
    assert json_has_phi(payload) is False
    assert "patient_id" not in dumped
    assert "doctor_fio" not in dumped
    assert "p-api-1" not in dumped
    assert "Иванов" not in dumped


def test_case_passport_and_summary_omit_visit_index(tmp_path: Path) -> None:
    warehouse, lab, pk = _setup(tmp_path)
    full = build_case_passport("v-2026", warehouse=warehouse, lab_path=lab)
    assert full["ok"] is True
    assert full["patient_key"] == pk
    assert full["case_id"] == "v-2026"
    summary = passport_summary_for_case("m-now", warehouse=warehouse, lab_path=lab)
    assert summary["ok"] is True
    assert "visits" not in summary
    assert "items" not in summary
    assert summary["coverage"]["n_visits"] == 2
    assert "2 визитов" in summary["context"]
    dumped = json_dumps_public(summary)
    assert json_has_phi(summary) is False
    assert pk in dumped
    assert "p-api-1" not in dumped


def test_labs_only_on_requested_date(tmp_path: Path) -> None:
    warehouse, lab, pk = _setup(tmp_path)
    empty = build_passport_labs(pk, warehouse=warehouse, lab_path=lab)
    assert empty["ok"] is True
    assert empty["items"] == []
    assert "2026-01-12" in empty["dates"]
    day = build_passport_labs(pk, day="2026-01-12", warehouse=warehouse, lab_path=lab)
    assert day["ok"] is True
    assert day["items"][0]["indicator_name"] == "Hb"
    assert day["items"][0]["value"] == "130"
    dumped = json_dumps_public(day)
    assert json_has_phi(day) is False
    assert "patient_id" not in dumped
    assert "p-api-1" not in dumped


def test_bad_key_rejected() -> None:
    assert build_patient_passport("not-a-hash")["error"] == "bad_patient_key"
    assert build_patient_passport("123")["error"] == "bad_patient_key"


def test_resolve_by_visit_id_omits_query_and_phi(tmp_path: Path) -> None:
    warehouse, _lab, pk = _setup(tmp_path)
    payload = resolve_patient_query("v-2026", warehouse=warehouse)
    assert payload["ok"] is True
    assert payload["patient_key"] == pk
    assert payload["latest_visit_id"] == "v-2026"
    assert "2 визитов" in payload["context"]
    dumped = json_dumps_public(payload)
    assert json_has_phi(payload) is False
    assert "p-api-1" not in dumped
    assert "patient_id" not in dumped
    assert resolve_patient_query("", warehouse=warehouse)["error"] == "empty_query"
    assert resolve_patient_query("missing-visit", warehouse=warehouse)["error"] == "passport_not_found"
