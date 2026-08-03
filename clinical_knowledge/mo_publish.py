"""Подготовка снапшота витрины МО для публикации в прод.

Витрину нельзя просто скопировать поверх продовой: в том же файле живут таблицы
кабинета методиста (`crm_case_state`, `crm_case_event`, `saved_view`, `export_job`),
которые заполняются **в проде** и локально пусты. Поэтому публикуем копию без CRM
и на стороне прода доливаем только факты и справочники.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from clinical_knowledge.mo_daily import CRM_TABLES, initialize_warehouse

# Таблицы, которыми владеет конвейер: их можно перезаливать целиком.
PUBLISHED_PREFIXES = ("fact_", "dim_")


def published_tables(db: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        if row[0].startswith(PUBLISHED_PREFIXES)
    ]


def build_publish_snapshot(warehouse: Path, target: Path) -> dict[str, int]:
    """Build a snapshot compatible with the currently deployed code schema.

    A development scorer can migrate the local warehouse before its server code
    reaches production. Copying that database byte-for-byte would make the
    remote ``INSERT ... SELECT *`` fail on different column counts. Start from
    the schema known to this checkout and copy only common pipeline columns.
    """
    if not warehouse.is_file():
        raise FileNotFoundError(f"нет витрины: {warehouse}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    initialize_warehouse(target)
    counts: dict[str, int] = {}
    with sqlite3.connect(target) as copy:
        copy.execute("ATTACH DATABASE ? AS source", (str(warehouse.resolve()),))
        source_tables = {
            row[0]
            for row in copy.execute("SELECT name FROM source.sqlite_master WHERE type='table'")
        }
        for table in published_tables(copy):
            if table not in source_tables:
                continue
            target_columns = [row[1] for row in copy.execute(f"PRAGMA main.table_info({table})")]
            source_columns = {
                row[1] for row in copy.execute(f"PRAGMA source.table_info({table})")
            }
            columns = [column for column in target_columns if column in source_columns]
            if not columns:
                continue
            quoted = ", ".join(f'"{column}"' for column in columns)
            copy.execute(
                f"INSERT OR REPLACE INTO main.{table} ({quoted}) "
                f"SELECT {quoted} FROM source.{table}"
            )
        copy.commit()
        copy.execute("DETACH DATABASE source")
        copy.execute("VACUUM")
        for table in published_tables(copy):
            counts[table] = copy.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


def merge_sql(tables: Iterable[str], *, snapshot_path: str) -> str:
    """SQL для прода: долить факты и справочники, не касаясь таблиц кабинета."""
    statements = [f"ATTACH DATABASE '{snapshot_path}' AS pub;", "BEGIN;"]
    for table in tables:
        statements.append(f"INSERT OR REPLACE INTO {table} SELECT * FROM pub.{table};")
    statements.extend(["COMMIT;", "DETACH DATABASE pub;"])
    return "\n".join(statements)


def snapshot_summary(snapshot: Path) -> dict[str, object]:
    """Что именно уезжает в прод: дни, дыры в оценках, границы периода."""
    with sqlite3.connect(snapshot) as db:
        days = db.execute("SELECT COUNT(*) FROM fact_mo_daily").fetchone()[0]
        first, last = db.execute("SELECT MIN(visit_date), MAX(visit_date) FROM fact_mo_daily").fetchone()
        cases, scored = db.execute(
            "SELECT COUNT(*), SUM(overall_pct IS NOT NULL) FROM fact_mo_case"
        ).fetchone()
        eligible_gaps = db.execute(
            """SELECT COUNT(*) FROM fact_mo_case
               WHERE overall_pct IS NULL AND document_kind IN ('medical_exam', 'consultation')"""
        ).fetchone()[0]
        crm_rows = sum(
            db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in CRM_TABLES
            if db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (table,)
            ).fetchone()
        )
    return {
        "days": days,
        "first_date": first,
        "last_date": last,
        "cases": cases,
        "scored": scored or 0,
        "eligible_without_score": eligible_gaps,
        "crm_rows": crm_rows,
    }
