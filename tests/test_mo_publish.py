"""Публикация витрины МО в прод не должна затирать работу кабинета методиста."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_daily import CRM_TABLES, initialize_warehouse  # noqa: E402
from clinical_knowledge.mo_publish import (  # noqa: E402
    build_publish_snapshot,
    common_columns,
    merge_sql,
    published_tables,
    snapshot_summary,
)


def _seed(path: Path) -> None:
    initialize_warehouse(path)
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO fact_mo_daily (visit_date, source_rows, scored_rows, avg_score, revision,"
            " quality_status, updated_at) VALUES ('2026-07-29', 10, 9, 87.0, 1, 'passed', 'now')"
        )
        db.execute(
            "INSERT INTO fact_mo_case (mis_id, visit_id, visit_date, document_kind, overall_pct,"
            " status, doctor_key, specialty, filial, content_hash, updated_at)"
            " VALUES ('1', '100', '2026-07-29', 'medical_exam', 87.0, 'good', 'ivanov', 'Терапевт',"
            " 'Филиал', 'hash', 'now')"
        )
        db.execute(
            "INSERT INTO crm_case_state (case_id, status, assignee, due_date, updated_at, updated_by)"
            " VALUES ('100', 'new', NULL, NULL, 'now', 'pipeline')"
        )
        db.commit()


def test_publish_snapshot_has_facts_and_no_crm_rows(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo_analytics.sqlite"
    _seed(warehouse)
    snapshot = tmp_path / "publish.sqlite"

    counts = build_publish_snapshot(warehouse, snapshot)

    assert counts["fact_mo_case"] == 1
    assert counts["fact_mo_daily"] == 1
    assert all(not table.startswith("crm_") for table in counts)
    summary = snapshot_summary(snapshot)
    assert summary["crm_rows"] == 0
    assert summary["last_date"] == "2026-07-29"
    assert summary["scored"] == 1
    # Локальная витрина не тронута: снапшот - копия.
    with sqlite3.connect(warehouse) as db:
        assert db.execute("SELECT COUNT(*) FROM crm_case_state").fetchone()[0] == 1


def test_publish_snapshot_ignores_schema_from_code_not_deployed_yet(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo_analytics.sqlite"
    _seed(warehouse)
    with sqlite3.connect(warehouse) as db:
        db.execute("ALTER TABLE fact_mo_case ADD COLUMN future_score REAL")
        db.execute("UPDATE fact_mo_case SET future_score = 99.0")
        db.execute("CREATE TABLE fact_future_scorer (case_id TEXT PRIMARY KEY, score REAL)")
        db.execute("INSERT INTO fact_future_scorer VALUES ('100', 99.0)")
        db.commit()

    snapshot = tmp_path / "publish.sqlite"
    counts = build_publish_snapshot(warehouse, snapshot)

    assert counts["fact_mo_case"] == 1
    assert "fact_future_scorer" not in counts
    with sqlite3.connect(snapshot) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(fact_mo_case)")}
        assert "future_score" not in columns


def test_merge_keeps_production_crm_state(tmp_path: Path) -> None:
    local = tmp_path / "local.sqlite"
    _seed(local)
    snapshot = tmp_path / "publish.sqlite"
    tables = sorted(build_publish_snapshot(local, snapshot))
    with sqlite3.connect(snapshot) as db:
        column_map = {
            table: [row[1] for row in db.execute(f'PRAGMA table_info("{table}")')]
            for table in tables
        }

    production = tmp_path / "prod.sqlite"
    initialize_warehouse(production)
    with sqlite3.connect(production) as db:
        db.execute(
            "INSERT INTO crm_case_state (case_id, status, assignee, due_date, updated_at, updated_by)"
            " VALUES ('100', 'in_review', 'Методист', '2026-07-31', 'now', 'ИП')"
        )
        db.commit()

    with sqlite3.connect(production) as db:
        db.executescript(
            merge_sql(tables, snapshot_path=str(snapshot), column_map=column_map)
        )

    with sqlite3.connect(production) as db:
        assert db.execute("SELECT COUNT(*) FROM fact_mo_case").fetchone()[0] == 1
        status, assignee = db.execute(
            "SELECT status, assignee FROM crm_case_state WHERE case_id = '100'"
        ).fetchone()
    assert (status, assignee) == ("in_review", "Методист")


def test_merge_named_columns_survives_legacy_column_order(tmp_path: Path) -> None:
    """Прод с ALTER-колонками в конце не должен получать сдвиг doctor_key."""
    snapshot = tmp_path / "snap.sqlite"
    initialize_warehouse(snapshot)
    with sqlite3.connect(snapshot) as db:
        db.execute(
            "INSERT INTO fact_mo_case (mis_id, visit_id, visit_date, document_kind, overall_pct,"
            " overall_pct_v3, status, scorer_version, score_schema_version, llm_cost_usd,"
            " doctor_key, specialty, filial, diagnosis_code, icd_chapter, content_hash, updated_at)"
            " VALUES ('9', '90', '2026-08-04', 'medical_exam', 77.6, 70.0, 'review', 'v4.0.0',"
            " '4.0', 0.0, 'abc123doctor', 'Офтальмолог', 'Захарова', 'H52.1', 'VII', 'hash', 'now')"
        )
        db.commit()

    production = tmp_path / "prod_legacy.sqlite"
    with sqlite3.connect(production) as db:
        db.executescript(
            """
            CREATE TABLE fact_mo_case (
              mis_id TEXT PRIMARY KEY, visit_id TEXT, visit_date TEXT NOT NULL,
              document_kind TEXT NOT NULL, overall_pct REAL, status TEXT,
              doctor_key TEXT, specialty TEXT, filial TEXT, content_hash TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            ALTER TABLE fact_mo_case ADD COLUMN diagnosis_code TEXT;
            ALTER TABLE fact_mo_case ADD COLUMN icd_chapter TEXT;
            ALTER TABLE fact_mo_case ADD COLUMN overall_pct_v3 REAL;
            ALTER TABLE fact_mo_case ADD COLUMN scorer_version TEXT;
            ALTER TABLE fact_mo_case ADD COLUMN score_schema_version TEXT;
            ALTER TABLE fact_mo_case ADD COLUMN llm_cost_usd REAL DEFAULT 0;
            ALTER TABLE fact_mo_case ADD COLUMN mkb_code_main_source TEXT;
            ALTER TABLE fact_mo_case ADD COLUMN mkb_code_main_slot TEXT;
            ALTER TABLE fact_mo_case ADD COLUMN patient_key TEXT;
            ALTER TABLE fact_mo_case ADD COLUMN doctor_id TEXT;
            ALTER TABLE fact_mo_case ADD COLUMN diagnosis_text TEXT;
            ALTER TABLE fact_mo_case ADD COLUMN history_prior_n INTEGER DEFAULT 0;
            ALTER TABLE fact_mo_case ADD COLUMN history_tier TEXT;
            """
        )
        # Как publish: только prod ∩ snapshot (новые zone_* появятся после DDL на проде).
        with sqlite3.connect(snapshot) as snap_db:
            columns = common_columns(db, snap_db, "fact_mo_case")
        assert "doctor_key" in columns
        assert "zone1_pct" not in columns
        db.executescript(
            merge_sql(
                ["fact_mo_case"],
                snapshot_path=str(snapshot),
                column_map={"fact_mo_case": columns},
            )
        )
        row = db.execute(
            "SELECT doctor_key, specialty, filial, status, overall_pct FROM fact_mo_case WHERE mis_id='9'"
        ).fetchone()
    assert row == ("abc123doctor", "Офтальмолог", "Захарова", "review", 77.6)


def test_merge_star_fails_when_prod_has_extra_finding_columns(tmp_path: Path) -> None:
    """Регрессия 2026-08-06: SELECT * при 15 vs 12 колонках fact_mo_finding."""
    snapshot = tmp_path / "snap.sqlite"
    with sqlite3.connect(snapshot) as db:
        db.executescript(
            """
            CREATE TABLE fact_mo_finding (
              mis_id TEXT NOT NULL, finding_code TEXT NOT NULL, severity TEXT,
              passed INTEGER, evidence TEXT, source_ref TEXT, axis TEXT,
              title_ru TEXT, detail_ru TEXT, trust_level TEXT,
              penalty_applied INTEGER DEFAULT 0, needs_human INTEGER DEFAULT 0,
              PRIMARY KEY (mis_id, finding_code)
            );
            """
        )
        keep = [row[1] for row in db.execute("PRAGMA table_info(fact_mo_finding)")]

    production = tmp_path / "prod.sqlite"
    initialize_warehouse(production)
    with sqlite3.connect(production) as db:
        prod_cols = [row[1] for row in db.execute("PRAGMA table_info(fact_mo_finding)")]
        assert "is_shadow" in prod_cols
        assert len(prod_cols) > len(keep)
    with sqlite3.connect(production) as db:
        with pytest.raises(sqlite3.Error):
            db.executescript(
                merge_sql(["fact_mo_finding"], snapshot_path=str(snapshot), column_map=None)
            )
    with sqlite3.connect(production) as db:
        intersect = [c for c in prod_cols if c in set(keep)]
        db.executescript(
            merge_sql(
                ["fact_mo_finding"],
                snapshot_path=str(snapshot),
                column_map={"fact_mo_finding": intersect},
            )
        )


def test_published_tables_cover_facts_and_dimensions(tmp_path: Path) -> None:
    warehouse = tmp_path / "mo_analytics.sqlite"
    initialize_warehouse(warehouse)
    with sqlite3.connect(warehouse) as db:
        tables = published_tables(db)
    assert {"fact_mo_case", "fact_mo_daily", "dim_doctor", "dim_date"} <= set(tables)
    assert not set(CRM_TABLES) & set(tables)


def test_missing_warehouse_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_publish_snapshot(tmp_path / "absent.sqlite", tmp_path / "publish.sqlite")
