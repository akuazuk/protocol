#!/usr/bin/env python3
"""Профиль склада МО в JSON (план 2026-09-26, волна T; разделы 2.1 и 2.1a).

Только агрегаты: покрытие по месяцам, распределения зон и оценок за месяц,
состав замечаний, очередь догрузки, даты сверок. Ни одной строки с PHI не
читается и не печатается. Запуск на GCE (read-only):

    sudo python3 scripts/ops/mo_warehouse_profile.py \
        --db /var/data/medical_exams/warehouse/mo_analytics.sqlite \
        --lab-db /var/data/medical_exams/warehouse/mo_lab.sqlite \
        --month 2026-09 --out /tmp/profile.json

Пороги плана (раздел 7): доля двух верхних значений overall_grade ≤ 75 %,
zone1=weak ≤ 40 %, zone2b=na ≤ 40 %, клинических месяцев с зонами = 9.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from typing import Any

DEFAULT_DB = "/var/data/medical_exams/warehouse/mo_analytics.sqlite"
DEFAULT_LAB_DB = "/var/data/medical_exams/warehouse/mo_lab.sqlite"


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
    except sqlite3.Error as err:
        return [{"_error": str(err)}]


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return bool(row)


def _month_bounds(month: str) -> tuple[str, str]:
    year, mon = (int(x) for x in month.split("-"))
    first = dt.date(year, mon, 1)
    nxt = dt.date(year + (mon == 12), 1 if mon == 12 else mon + 1, 1)
    return first.isoformat(), (nxt - dt.timedelta(days=1)).isoformat()


def profile(db: str, lab_db: str | None, month: str) -> dict[str, Any]:
    date_from, date_to = _month_bounds(month)
    out: dict[str, Any] = {
        "taken_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "db": db,
        "month": month,
    }
    with _connect(db) as conn:
        out["coverage_by_month"] = _rows(conn, """
            SELECT substr(visit_date,1,7) AS month,
                   COUNT(*) AS cases,
                   SUM(document_kind='clinical_visit') AS clinical,
                   SUM(document_kind='clinical_visit' AND zone1_band IS NOT NULL AND zone1_band<>'') AS clinical_scored,
                   SUM(diagnosis_text IS NOT NULL AND diagnosis_text<>'') AS with_dx_text
            FROM fact_mo_case
            WHERE visit_date >= '2025-12-01'
            GROUP BY 1 ORDER BY 1""")
        out["document_kinds_month"] = _rows(conn, """
            SELECT document_kind, COUNT(*) AS n FROM fact_mo_case
            WHERE visit_date BETWEEN ? AND ? GROUP BY 1 ORDER BY n DESC""", (date_from, date_to))
        clinical_where = "visit_date BETWEEN ? AND ? AND document_kind='clinical_visit'"
        for col in ("overall_grade", "zone1_band", "zone2a_band", "zone2b_band", "zone2b_kp_status", "attention_primary", "reg55_band"):
            out[f"dist_{col}"] = _rows(conn, f"""
                SELECT COALESCE(NULLIF({col},''),'(null)') AS value, COUNT(*) AS n
                FROM fact_mo_case WHERE {clinical_where} GROUP BY 1 ORDER BY n DESC""", (date_from, date_to))
        out["no_icd_month"] = _rows(conn, f"""
            SELECT SUM(diagnosis_code IS NULL OR diagnosis_code='') AS without_code, COUNT(*) AS clinical
            FROM fact_mo_case WHERE {clinical_where}""", (date_from, date_to))
        if _has_table(conn, "fact_mo_finding"):
            out["findings_top"] = _rows(conn, """
                SELECT f.finding_code, f.severity, COUNT(DISTINCT f.mis_id) AS cases
                FROM fact_mo_finding f JOIN fact_mo_case c ON c.mis_id = f.mis_id
                WHERE c.visit_date BETWEEN ? AND ? AND c.document_kind='clinical_visit'
                GROUP BY 1,2 ORDER BY cases DESC LIMIT 40""", (date_from, date_to))
        if _has_table(conn, "dim_diagnosis"):
            out["dim_diagnosis_rows"] = _rows(conn, "SELECT COUNT(*) AS n FROM dim_diagnosis")
        out["distinct_dx_text_month"] = _rows(conn, f"""
            SELECT COUNT(DISTINCT diagnosis_text) AS n FROM fact_mo_case WHERE {clinical_where}""", (date_from, date_to))
        if _has_table(conn, "mo_ingest_job"):
            out["ingest_jobs"] = _rows(conn, """
                SELECT status, COUNT(*) AS n, MAX(COALESCE(updated_at, created_at)) AS last_change
                FROM mo_ingest_job GROUP BY 1""")
        if _has_table(conn, "fact_mis_catalog"):
            out["mis_catalog"] = _rows(conn, """
                SELECT COUNT(*) AS visits, MIN(visit_date) AS first, MAX(visit_date) AS last
                FROM fact_mis_catalog""")
        for table in ("crm_case_state", "crm_review_pack", "saved_view", "export_job", "access_log"):
            if _has_table(conn, table):
                out[f"rows_{table}"] = _rows(conn, f"SELECT COUNT(*) AS n FROM {table}")
    if lab_db:
        try:
            with _connect(lab_db) as lab:
                out["lab_by_month"] = _rows(lab, """
                    SELECT substr(COALESCE(result_date, sample_date, visit_date),1,7) AS month, COUNT(*) AS rows_n
                    FROM fact_mo_lab GROUP BY 1 ORDER BY 1""")
                if out["lab_by_month"] and "_error" in out["lab_by_month"][0]:
                    out["lab_by_month"] = _rows(lab, "SELECT COUNT(*) AS rows_n FROM fact_mo_lab")
        except sqlite3.Error as err:
            out["lab_error"] = str(err)

    # Метрики плана (раздел 7).
    def share(dist_key: str, values: tuple[str, ...]) -> float | None:
        dist = out.get(dist_key) or []
        total = sum(int(r.get("n") or 0) for r in dist if "n" in r)
        if not total:
            return None
        part = sum(int(r["n"]) for r in dist if r.get("value") in values)
        return round(100.0 * part / total, 1)

    grades = sorted((int(r["n"]) for r in out.get("dist_overall_grade", []) if "n" in r), reverse=True)
    grade_total = sum(grades)
    out["metrics"] = {
        "top2_overall_grade_share_pct": round(100.0 * sum(grades[:2]) / grade_total, 1) if grade_total else None,
        "zone1_weak_pct": share("dist_zone1_band", ("weak",)),
        "zone1_ok_pct": share("dist_zone1_band", ("ok",)),
        "zone2b_na_pct": share("dist_zone2b_band", ("na", "(null)")),
        "zone2b_ok_pct": share("dist_zone2b_band", ("ok",)),
        "kp_matched_pct": share("dist_zone2b_kp_status", ("matched",)),
        "clinical_months_with_zones": sum(
            1 for r in out.get("coverage_by_month", []) if str(r.get("month", "")).startswith("2026") and int(r.get("clinical_scored") or 0) > 0
        ),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--lab-db", default=DEFAULT_LAB_DB)
    parser.add_argument("--no-lab", action="store_true")
    parser.add_argument("--month", default=dt.date.today().strftime("%Y-%m"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    doc = profile(args.db, None if args.no_lab else args.lab_db, args.month)
    text = json.dumps(doc, ensure_ascii=False, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(json.dumps(doc["metrics"], ensure_ascii=False))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
