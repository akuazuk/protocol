"""P0: паспорт клиента со склада, без PHI."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from clinical_knowledge.mo_daily import initialize_warehouse, patient_key_for
from clinical_knowledge.mo_patient_passport import (
    ENGINE,
    public_passport,
    rebuild_passports,
)


def _insert_case(
    db: sqlite3.Connection,
    *,
    mis_id: str,
    visit_id: str,
    patient_key: str,
    visit_date: str,
    specialty: str,
    kind: str = "clinical_visit",
    code: str = "J03.9",
    text: str = "Острый тонзиллит",
    grade: str = "fair",
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
            kind,
            70.0,
            "doc-a",
            specialty,
            patient_key,
            code,
            text,
            grade,
            mis_id,
            "2026-09-27T00:00:00Z",
        ),
    )


def test_two_kz_same_visit_collapse_to_one_card(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo_analytics.sqlite"
    initialize_warehouse(warehouse)
    pk = patient_key_for("p-passport-1")
    with sqlite3.connect(warehouse) as db:
        _insert_case(
            db,
            mis_id="m1",
            visit_id="v100",
            patient_key=pk,
            visit_date="2026-04-02",
            specialty="Терапевт",
            kind="consultation",
            grade="poor",
        )
        _insert_case(
            db,
            mis_id="m2",
            visit_id="v100",
            patient_key=pk,
            visit_date="2026-04-02",
            specialty="Терапевт",
            kind="clinical_visit",
            grade="fair",
        )
        db.commit()
    out = rebuild_passports(warehouse)
    assert out["ok"] is True
    assert out["engine"] == ENGINE
    assert out["passports"] == 1
    assert out["visit_cards"] == 1
    with sqlite3.connect(warehouse) as db:
        row = db.execute(
            "SELECT document_kind, overall_grade, mis_id FROM fact_mo_visit_index"
        ).fetchone()
        assert row[0] == "clinical_visit"
        assert row[1] == "fair"
        assert row[2] == "m2"


def test_same_visit_id_two_keys_stay_two_cards(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo_analytics.sqlite"
    initialize_warehouse(warehouse)
    a = patient_key_for("p-a")
    b = patient_key_for("p-b")
    with sqlite3.connect(warehouse) as db:
        _insert_case(db, mis_id="a1", visit_id="shared", patient_key=a, visit_date="2026-03-01", specialty="ЛОР")
        _insert_case(db, mis_id="b1", visit_id="shared", patient_key=b, visit_date="2026-03-01", specialty="ЛОР")
        db.commit()
    out = rebuild_passports(warehouse)
    assert out["visit_cards"] == 2
    assert out["passports"] == 2


def test_specialty_tiles_and_lab_counts(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo_analytics.sqlite"
    lab = tmp_path / "mo_lab.sqlite"
    initialize_warehouse(warehouse)
    pk = patient_key_for("p-labs")
    with sqlite3.connect(warehouse) as db:
        _insert_case(db, mis_id="t1", visit_id="v1", patient_key=pk, visit_date="2026-01-10", specialty="Терапевт")
        _insert_case(db, mis_id="l1", visit_id="v2", patient_key=pk, visit_date="2026-02-11", specialty="ЛОР", code="J32.0")
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
        db.executemany(
            "INSERT INTO fact_mo_lab VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (pk, "2026-01-12", 1, 10, "ОАК", 1, "Hb", "130", "г/л"),
                (pk, "2026-02-01", 2, 10, "ОАК", 1, "Hb", "128", "г/л"),
            ],
        )
        db.commit()
    rebuild_passports(warehouse, lab_path=lab)
    with sqlite3.connect(warehouse) as db:
        spec_n = db.execute("SELECT COUNT(*) FROM fact_mo_patient_specialty").fetchone()[0]
        assert spec_n == 2
        passport = dict(
            zip(
                [
                    "patient_key",
                    "n_visits",
                    "n_specialties",
                    "first_date",
                    "last_date",
                    "n_lab_rows",
                    "n_lab_dates",
                ],
                db.execute(
                    """SELECT patient_key, n_visits, n_specialties, first_date, last_date,
                              n_lab_rows, n_lab_dates
                       FROM fact_mo_patient_passport"""
                ).fetchone(),
            )
        )
    assert passport["n_visits"] == 2
    assert passport["n_specialties"] == 2
    assert passport["first_date"] == "2026-01-10"
    assert passport["last_date"] == "2026-02-11"
    assert passport["n_lab_rows"] == 2
    assert passport["n_lab_dates"] == 2
    dumped = json.dumps(public_passport(passport), ensure_ascii=False)
    assert "patient_id" not in dumped
    assert pk not in dumped
    assert "p-labs" not in dumped


def test_incremental_rebuild_only_touched_key(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo_analytics.sqlite"
    initialize_warehouse(warehouse)
    a = patient_key_for("keep")
    b = patient_key_for("touch")
    with sqlite3.connect(warehouse) as db:
        _insert_case(db, mis_id="a1", visit_id="va", patient_key=a, visit_date="2026-05-01", specialty="Терапевт")
        _insert_case(db, mis_id="b1", visit_id="vb", patient_key=b, visit_date="2026-05-02", specialty="Терапевт")
        db.commit()
    rebuild_passports(warehouse)
    with sqlite3.connect(warehouse) as db:
        _insert_case(db, mis_id="b2", visit_id="vb2", patient_key=b, visit_date="2026-06-01", specialty="Хирург")
        db.commit()
    rebuild_passports(warehouse, patient_keys=[b])
    with sqlite3.connect(warehouse) as db:
        n_a = db.execute(
            "SELECT n_visits FROM fact_mo_patient_passport WHERE patient_key=?", (a,)
        ).fetchone()[0]
        n_b = db.execute(
            "SELECT n_visits, n_specialties FROM fact_mo_patient_passport WHERE patient_key=?",
            (b,),
        ).fetchone()
    assert n_a == 1
    assert n_b[0] == 2
    assert n_b[1] == 2
