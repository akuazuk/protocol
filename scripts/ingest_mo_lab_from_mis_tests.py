#!/usr/bin/env python3
"""GCE: mis_tests -> warehouse/mo_lab.sqlite (patient_key, без сырого patient_id).

Канон: source /opt/protocol/deploy/gcp-app/load_mis_env.sh
Значения анализов на диск склада; в stdout только счётчики, не PHI.

  PYTHONPATH=/opt/protocol python3 scripts/ingest_mo_lab_from_mis_tests.py \
    --from 2025-12-19 --to 2026-08-26
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DDL = """
CREATE TABLE IF NOT EXISTS fact_mo_lab (
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
CREATE INDEX IF NOT EXISTS idx_lab_patient_date ON fact_mo_lab(patient_key, test_date);
CREATE INDEX IF NOT EXISTS idx_lab_date ON fact_mo_lab(test_date);
CREATE TABLE IF NOT EXISTS fact_mo_lab_meta (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""

SELECT_SQL = """
SELECT date, patient_id, test_id, type_id, type_name,
       indicator_id, indicator_name, value, unit
FROM mis_tests
WHERE date >= %s AND date < %s
"""


def patient_key_for(patient_id: object) -> str:
    """Тот же hash, что clinical_knowledge.mo_daily.patient_key_for."""
    normalized = str(patient_id or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20] if normalized else ""


def _month_starts(d0: date, d1: date) -> list[tuple[date, date]]:
    """Полуинтервалы [start, end) по календарным месяцам внутри [d0, d1)."""
    out: list[tuple[date, date]] = []
    cur = date(d0.year, d0.month, 1)
    if d0.day != 1:
        cur = d0
    while cur < d1:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        chunk_end = min(nxt, d1)
        out.append((cur, chunk_end))
        cur = chunk_end
    return out


def _connect_mis():
    import pymysql

    pw = (os.environ.get("KRAVIRA_DB_PASSWORD") or "").strip()
    if not pw:
        raise SystemExit("Нет KRAVIRA_DB_PASSWORD (GCE load_mis_env.sh)")
    return pymysql.connect(
        host=(os.environ.get("KRAVIRA_DB_HOST") or "178.163.240.131").strip(),
        port=int(os.environ.get("KRAVIRA_DB_PORT") or "6330"),
        user=(os.environ.get("KRAVIRA_DB_USER") or "kravira_mc_user").strip(),
        password=pw,
        database=(os.environ.get("KRAVIRA_DB_NAME") or "kravira_mc").strip(),
        charset="utf8mb4",
        connect_timeout=int(os.environ.get("MIS_DB_CONNECT_TIMEOUT", "30")),
        read_timeout=int(os.environ.get("MIS_DB_READ_TIMEOUT", "600")),
        cursorclass=pymysql.cursors.SSCursor,
    )


def ingest(out: Path, date_from: date, date_to_exclusive: date) -> dict[str, object]:
    out.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(out))
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(DDL)
    db.execute(
        "DELETE FROM fact_mo_lab WHERE test_date >= ? AND test_date < ?",
        (date_from.isoformat(), date_to_exclusive.isoformat()),
    )
    inserted = 0
    skipped = 0
    mis = _connect_mis()
    try:
        for start, end in _month_starts(date_from, date_to_exclusive):
            cur = mis.cursor()
            cur.execute(SELECT_SQL, (start.isoformat(), end.isoformat()))
            batch: list[tuple] = []
            while True:
                rows = cur.fetchmany(5000)
                if not rows:
                    break
                for row in rows:
                    key = patient_key_for(row[1])
                    if not key:
                        skipped += 1
                        continue
                    test_date = row[0]
                    if hasattr(test_date, "isoformat"):
                        test_date = test_date.isoformat()
                    batch.append(
                        (
                            key,
                            str(test_date),
                            int(row[2] or 0),
                            int(row[3] or 0),
                            str(row[4] or ""),
                            int(row[5] or 0),
                            str(row[6] or ""),
                            str(row[7] or ""),
                            str(row[8] or ""),
                        )
                    )
                if batch:
                    db.executemany(
                        "INSERT INTO fact_mo_lab VALUES (?,?,?,?,?,?,?,?,?)",
                        batch,
                    )
                    inserted += len(batch)
                    batch = []
                    db.commit()
            cur.close()
            print(
                f"chunk {start.isoformat()}..{end.isoformat()} inserted={inserted}",
                flush=True,
            )
    finally:
        mis.close()
    n = db.execute("SELECT COUNT(*) FROM fact_mo_lab").fetchone()[0]
    dmin, dmax = db.execute(
        "SELECT MIN(test_date), MAX(test_date) FROM fact_mo_lab"
    ).fetchone()
    n_keys = db.execute(
        "SELECT COUNT(DISTINCT patient_key) FROM fact_mo_lab"
    ).fetchone()[0]
    n_types = db.execute(
        "SELECT COUNT(DISTINCT type_id) FROM fact_mo_lab"
    ).fetchone()[0]
    meta = {
        "ok": True,
        "engine": "ingest_mo_lab_from_mis_tests_v1",
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "date_from": date_from.isoformat(),
        "date_to_exclusive": date_to_exclusive.isoformat(),
        "rows": n,
        "inserted_this_run": inserted,
        "skipped_empty_patient": skipped,
        "patient_keys": n_keys,
        "type_ids": n_types,
        "test_date_min": dmin,
        "test_date_max": dmax,
        "path": str(out),
    }
    db.execute(
        "INSERT OR REPLACE INTO fact_mo_lab_meta(key, value) VALUES ('last_ingest', ?)",
        (json.dumps(meta, ensure_ascii=False),),
    )
    db.commit()
    db.close()
    return meta


def coverage_vs_cases(lab_path: Path, cases_path: Path) -> dict[str, int]:
    """Пересечение patient_key без выгрузки ключей."""
    if not cases_path.is_file():
        return {}
    lab = sqlite3.connect(f"file:{lab_path}?mode=ro", uri=True)
    attach = f"file:{cases_path}?mode=ro"
    try:
        lab.execute("ATTACH DATABASE ? AS casesdb", (attach,))
    except sqlite3.OperationalError:
        lab.execute("ATTACH DATABASE ? AS casesdb", (str(cases_path),))
    n_case_keys = lab.execute(
        "SELECT COUNT(DISTINCT patient_key) FROM casesdb.fact_mo_case "
        "WHERE COALESCE(patient_key,'') != ''"
    ).fetchone()[0]
    n_overlap = lab.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT DISTINCT l.patient_key
          FROM fact_mo_lab l
          INNER JOIN casesdb.fact_mo_case c ON c.patient_key = l.patient_key
          WHERE COALESCE(c.patient_key,'') != ''
        )
        """
    ).fetchone()[0]
    n_case_with_lab_window = lab.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT DISTINCT c.rowid
          FROM casesdb.fact_mo_case c
          INNER JOIN fact_mo_lab l ON l.patient_key = c.patient_key
            AND l.test_date BETWEEN date(c.visit_date, '-14 day')
                                AND date(c.visit_date, '+1 day')
          WHERE COALESCE(c.patient_key,'') != ''
        )
        """
    ).fetchone()[0]
    n_cases = lab.execute("SELECT COUNT(*) FROM casesdb.fact_mo_case").fetchone()[0]
    lab.close()
    return {
        "case_rows": n_cases,
        "case_patient_keys": n_case_keys,
        "lab_keys_in_cases": n_overlap,
        "case_rows_with_lab_window": n_case_with_lab_window,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="date_from", default="2025-12-19")
    ap.add_argument("--to", dest="date_to", default="2026-08-26", help="exclusive")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/var/data/medical_exams/warehouse/mo_lab.sqlite"),
    )
    ap.add_argument(
        "--cases-db",
        type=Path,
        default=Path("/var/data/medical_exams/warehouse/mo_analytics.sqlite"),
    )
    args = ap.parse_args()
    d0 = date.fromisoformat(args.date_from)
    d1 = date.fromisoformat(args.date_to)
    if d1 <= d0:
        raise SystemExit("--to must be exclusive and after --from")
    meta = ingest(args.out.expanduser(), d0, d1)
    print(json.dumps({k: meta[k] for k in (
        "ok", "rows", "inserted_this_run", "patient_keys", "type_ids",
        "test_date_min", "test_date_max", "date_from", "date_to_exclusive",
    )}, ensure_ascii=False))
    cov_path = args.cases_db.expanduser()
    if cov_path.is_file():
        try:
            cov = coverage_vs_cases(args.out.expanduser(), cov_path)
        except sqlite3.OperationalError as exc:
            print(json.dumps({"coverage_error": str(exc)}, ensure_ascii=False))
            cov = {}
        if cov:
            print(json.dumps({"coverage": cov}, ensure_ascii=False))
            meta["coverage"] = cov
        report = Path("/var/data/medical_exams/reports/mo_lab_ingest_meta.json")
        try:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
