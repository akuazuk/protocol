"""Lab bundle for case review: patient_key + date window, no foreign rows."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from clinical_knowledge.mo_daily import patient_key_for
from clinical_knowledge.mo_lab_bundle import (
    ENGINE,
    attach_lab_to_case,
    build_lab_bundle,
    lab_payload_for_case,
    public_lab_for_ui,
)

DDL = """
CREATE TABLE fact_mo_lab (
  patient_key TEXT NOT NULL,
  test_date TEXT NOT NULL,
  test_id INTEGER NOT NULL,
  type_id INTEGER,
  type_name TEXT,
  indicator_id INTEGER,
  indicator_name TEXT,
  value TEXT,
  unit TEXT
);
"""


def _seed(path: Path) -> str:
    pk = patient_key_for("1001")
    db = sqlite3.connect(path)
    db.executescript(DDL)
    rows = [
        (pk, "2026-08-10", 1, 10, "ОАК", 101, "Гемоглобин", "132", "г/л"),
        (pk, "2026-08-10", 1, 10, "ОАК", 102, "Лейкоциты", "6.1", "10^9/л"),
        (pk, "2026-08-20", 2, 20, "БАК", 201, "Глюкоза", "5.4", "ммоль/л"),
        (pk, "2026-07-01", 3, 10, "ОАК", 101, "Гемоглобин", "140", "г/л"),
        (patient_key_for("9999"), "2026-08-20", 4, 10, "ОАК", 101, "Чужой", "1", ""),
    ]
    db.executemany("INSERT INTO fact_mo_lab VALUES (?,?,?,?,?,?,?,?,?)", rows)
    db.commit()
    db.close()
    return pk


def test_window_groups_types_and_excludes_foreign(tmp_path: Path) -> None:
    db = tmp_path / "mo_lab.sqlite"
    pk = _seed(db)
    bundle = build_lab_bundle(
        patient_key=pk,
        visit_date="2026-08-20",
        lab_db=db,
    )
    assert bundle["engine"] == ENGINE
    assert bundle["reason"] == ""
    assert bundle["summary"]["n_rows"] == 3
    assert bundle["summary"]["n_dates"] == 2
    assert bundle["summary"]["n_types"] == 2
    assert bundle["summary"]["same_day_rows"] == 1
    dates = [day["test_date"] for day in bundle["days"]]
    assert dates == ["2026-08-20", "2026-08-10"]
    same_day = bundle["days"][0]
    assert same_day["same_day"] is True
    assert same_day["types"][0]["type_name"] == "БАК"
    names = {ind["name"] for ind in bundle["days"][1]["types"][0]["indicators"]}
    assert names == {"Гемоглобин", "Лейкоциты"}
    blob = str(bundle)
    assert "Чужой" not in blob
    assert "1001" not in blob
    pub = str(public_lab_for_ui(bundle))
    assert pk not in pub


def test_empty_without_key_is_honest(tmp_path: Path) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    bundle = build_lab_bundle(patient_key="", visit_date="2026-08-20", lab_db=db)
    assert bundle["reason"] == "missing_key"
    assert bundle["summary"]["n_rows"] == 0


def test_empty_window_reason(tmp_path: Path) -> None:
    db = tmp_path / "mo_lab.sqlite"
    pk = _seed(db)
    bundle = build_lab_bundle(
        patient_key=pk,
        visit_date="2026-01-15",
        lab_db=db,
    )
    assert bundle["reason"] == "empty"
    assert bundle["window"]["from"] == "2026-01-01"


def test_attach_hashes_patient_id(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    monkeypatch.delenv("MO_LAB_DB", raising=False)
    case = {"patient_id": "1001", "visit_date": "2026-08-20"}
    payload = lab_payload_for_case(case, lab_db=db)
    assert payload["summary"]["n_rows"] == 3
    assert "patient_id" not in payload
    assert "patient_key" not in payload
    assert case["_lab"]["summary"]["n_rows"] == 3


def test_flag_off_skips_query(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "mo_lab.sqlite"
    pk = _seed(db)
    monkeypatch.setenv("MO_LAB_BUNDLE", "0")
    bundle = build_lab_bundle(patient_key=pk, visit_date="2026-08-20", lab_db=db)
    assert bundle["reason"] == "disabled"
    assert bundle["summary"]["n_rows"] == 0


def test_public_ui_has_no_identity(tmp_path: Path) -> None:
    db = tmp_path / "mo_lab.sqlite"
    _seed(db)
    case = {"patient_id": "1001", "patient_key": patient_key_for("1001"), "date": "2026-08-20"}
    attach_lab_to_case(case, lab_db=db)
    pub = public_lab_for_ui(case["_lab"])
    assert pub["engine"] == ENGINE
    dumped = str(pub)
    assert "1001" not in dumped
    assert patient_key_for("1001") not in dumped


def test_exact_row_cap_is_not_reported_as_truncated(tmp_path: Path) -> None:
    db = tmp_path / "mo_lab.sqlite"
    pk = _seed(db)
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "INSERT INTO fact_mo_lab VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    pk,
                    "2026-08-20",
                    1000 + idx,
                    90,
                    "Прочее",
                    1000 + idx,
                    f"Показатель {idx}",
                    str(idx),
                    "",
                )
                for idx in range(397)
            ],
        )
    bundle = build_lab_bundle(
        patient_key=pk,
        visit_date="2026-08-20",
        lab_db=db,
    )
    assert bundle["summary"]["n_rows"] == 400
    assert bundle["summary"]["truncated"] is False
