#!/usr/bin/env python3
"""Проставить patient_key на складе из MIS (только id/date/visit_id/patient_id).

Канон: GCE, пароль из Secret Manager / .env.mis. Сырой patient_id в логи не пишем.
Не импортирует clinical_knowledge на старте: venv-mis без pydantic.

  source /opt/protocol/deploy/gcp-app/load_mis_env.sh
  PYTHONPATH=/opt/protocol python3 /opt/protocol/scripts/backfill_mo_patient_keys_from_mis.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def patient_key_for(patient_id: object) -> str:
    normalized = str(patient_id or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20] if normalized else ""


def apply_keys(
    db: sqlite3.Connection,
    *,
    by_visit: dict[str, str],
    by_mis: dict[str, str],
) -> dict[str, int]:
    db.execute("DROP TABLE IF EXISTS tmp_key_visit")
    db.execute("DROP TABLE IF EXISTS tmp_key_mis")
    db.execute(
        "CREATE TEMP TABLE tmp_key_visit (visit_id TEXT PRIMARY KEY, patient_key TEXT NOT NULL)"
    )
    db.execute(
        "CREATE TEMP TABLE tmp_key_mis (mis_id TEXT PRIMARY KEY, patient_key TEXT NOT NULL)"
    )
    if by_visit:
        db.executemany(
            "INSERT OR REPLACE INTO tmp_key_visit VALUES (?,?)",
            by_visit.items(),
        )
    if by_mis:
        db.executemany(
            "INSERT OR REPLACE INTO tmp_key_mis VALUES (?,?)",
            by_mis.items(),
        )
    cur_visit = db.execute(
        """
        UPDATE fact_mo_case
           SET patient_key = (
                SELECT patient_key FROM tmp_key_visit
                 WHERE tmp_key_visit.visit_id = fact_mo_case.visit_id
           )
         WHERE length(trim(coalesce(patient_key, ''))) = 0
           AND visit_id IN (SELECT visit_id FROM tmp_key_visit)
        """
    )
    cur_mis = db.execute(
        """
        UPDATE fact_mo_case
           SET patient_key = (
                SELECT patient_key FROM tmp_key_mis
                 WHERE tmp_key_mis.mis_id = fact_mo_case.mis_id
           )
         WHERE length(trim(coalesce(patient_key, ''))) = 0
           AND mis_id IN (SELECT mis_id FROM tmp_key_mis)
        """
    )
    db.commit()
    return {
        "updated_by_visit": int(cur_visit.rowcount or 0),
        "updated_by_mis": int(cur_mis.rowcount or 0),
    }


def _fetch_identity(date_from: str, date_to: str) -> list[tuple[str, str, str, str]]:
    import pymysql

    pw = (os.environ.get("KRAVIRA_DB_PASSWORD") or "").strip()
    if not pw:
        raise SystemExit("no_mis_password")
    con = pymysql.connect(
        host=os.environ.get("KRAVIRA_DB_HOST") or "178.163.240.131",
        port=int(os.environ.get("KRAVIRA_DB_PORT") or 6330),
        user=os.environ.get("KRAVIRA_DB_USER") or "kravira_mc_user",
        password=pw,
        database=os.environ.get("KRAVIRA_DB_NAME") or "kravira_mc",
        charset="utf8mb4",
        connect_timeout=30,
        read_timeout=600,
    )
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, date, visit_id, patient_id
            FROM mis_protocol
            WHERE date >= %s AND date < %s
              AND patient_id IS NOT NULL AND CAST(patient_id AS CHAR) <> ''
            """,
            (date_from, date_to),
        )
        rows = []
        for mid, day, visit_id, patient_id in cur.fetchall():
            day_s = day.isoformat() if hasattr(day, "isoformat") else str(day)[:10]
            rows.append((str(mid), day_s, str(visit_id or ""), str(patient_id or "").strip()))
        return rows
    finally:
        con.close()


def _refresh_history(db: sqlite3.Connection, *, since: str = "") -> dict[str, int]:
    try:
        from clinical_knowledge.mo_patient_history_bundle import build_patient_history_bundle
    except ImportError:
        print("history_refresh_skipped no_clinical_knowledge_deps", flush=True)
        return {"refreshed": 0, "skipped": 1}

    cols = {row[1] for row in db.execute("PRAGMA table_info(fact_mo_case)")}
    if "patient_key" not in cols or "history_prior_n" not in cols:
        return {"refreshed": 0}
    sql = """
        SELECT mis_id, visit_id, visit_date, patient_key, doctor_key, doctor_id,
               specialty, diagnosis_code
        FROM fact_mo_case
        WHERE TRIM(COALESCE(patient_key,'')) != ''
    """
    params: tuple[str, ...] = ()
    if since:
        sql += " AND visit_date >= ?"
        params = (since,)
    sql += " ORDER BY visit_date, mis_id"
    keyed = db.execute(sql, params).fetchall()
    updated = 0
    for idx, row in enumerate(keyed, 1):
        day = str(row["visit_date"] or "")[:10]
        bundle = build_patient_history_bundle(
            patient_key=str(row["patient_key"] or ""),
            as_of_date=day,
            doctor_id=str(row["doctor_id"] or "") if "doctor_id" in row.keys() else "",
            doctor_key=str(row["doctor_key"] or ""),
            specialty=str(row["specialty"] or ""),
            current_code=str(row["diagnosis_code"] or ""),
            exclude_ids={str(row["mis_id"] or ""), str(row["visit_id"] or "")},
            warehouse=db,
        )
        summary = bundle.get("summary") or {}
        prior_n = int(summary.get("n_visits") or 0)
        tier = str(bundle.get("tier") or "")
        db.execute(
            """UPDATE fact_mo_case
               SET history_prior_n=?, history_tier=?
               WHERE mis_id=?""",
            (prior_n, tier, row["mis_id"]),
        )
        updated += 1
        if idx % 2000 == 0:
            db.commit()
            print("history_refresh_progress", idx, flush=True)
    db.commit()
    return {"refreshed": updated}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", default="2026-01-01")
    parser.add_argument("--to", dest="date_to", default="2026-07-26")
    parser.add_argument(
        "--warehouse",
        default=os.environ.get("MO_WAREHOUSE")
        or "/var/data/medical_exams/warehouse/mo_analytics.sqlite",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-history-refresh", action="store_true")
    parser.add_argument("--refresh-since", default="")
    args = parser.parse_args()
    date.fromisoformat(args.date_from)
    date.fromisoformat(args.date_to)
    if args.refresh_since:
        date.fromisoformat(args.refresh_since)

    ident = _fetch_identity(args.date_from, args.date_to)
    print("mis_identity_rows", len(ident), flush=True)
    by_visit: dict[str, str] = {}
    by_mis: dict[str, str] = {}
    for mid, _day, visit_id, patient_id in ident:
        key = patient_key_for(patient_id)
        if not key:
            continue
        if visit_id:
            by_visit[visit_id] = key
        if mid:
            by_mis[mid] = key
    print("keys_from_mis", len(set(by_visit.values()) | set(by_mis.values())), flush=True)

    path = Path(args.warehouse)
    if not path.is_file():
        raise SystemExit(f"no_warehouse:{path}")
    db = sqlite3.connect(str(path))
    db.row_factory = sqlite3.Row
    updated = {"updated_by_visit": 0, "updated_by_mis": 0}
    if not args.dry_run:
        updated = apply_keys(db, by_visit=by_visit, by_mis=by_mis)
    keyed = db.execute(
        "SELECT COUNT(*) FROM fact_mo_case WHERE TRIM(COALESCE(patient_key,''))!=''"
    ).fetchone()[0]
    print(
        "warehouse_updated",
        updated,
        "keyed_now",
        keyed,
        flush=True,
    )
    if not args.dry_run and not args.skip_history_refresh:
        stats = _refresh_history(db, since=args.refresh_since)
        print("history_refresh", stats, flush=True)
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
