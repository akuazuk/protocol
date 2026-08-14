#!/usr/bin/env python3
"""Проставить patient_key на складе из MIS (только id/date/visit_id/patient_id).

Канон: GCE, пароль из Secret Manager / .env.mis. Сырой patient_id в логи не пишем.

  source /opt/protocol/deploy/gcp-app/load_mis_env.sh
  PYTHONPATH=/opt/protocol python3 /opt/protocol/scripts/backfill_mo_patient_keys_from_mis.py
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_daily import patient_key_for  # noqa: E402
from clinical_knowledge.mo_patient_history_bundle import build_patient_history_bundle  # noqa: E402


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


def _refresh_history(db: sqlite3.Connection) -> dict[str, int]:
    cols = {row[1] for row in db.execute("PRAGMA table_info(fact_mo_case)")}
    if "patient_key" not in cols or "history_prior_n" not in cols:
        return {"refreshed": 0}
    keyed = db.execute(
        """
        SELECT mis_id, visit_id, visit_date, patient_key, doctor_key, doctor_id,
               specialty, diagnosis_code
        FROM fact_mo_case
        WHERE TRIM(COALESCE(patient_key,'')) != ''
        ORDER BY patient_key, visit_date, mis_id
        """
    ).fetchall()
    by_key: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in keyed:
        by_key[str(row["patient_key"])].append(row)
    updated = 0
    for key, rows in by_key.items():
        for idx, row in enumerate(rows):
            prior_n = idx
            day = str(row["visit_date"] or "")[:10]
            bundle = build_patient_history_bundle(
                patient_key=key,
                as_of_date=day,
                doctor_id=str(row["doctor_id"] or "") if "doctor_id" in row.keys() else "",
                doctor_key=str(row["doctor_key"] or ""),
                specialty=str(row["specialty"] or ""),
                current_code=str(row["diagnosis_code"] or ""),
                exclude_ids={str(row["mis_id"] or ""), str(row["visit_id"] or "")},
                warehouse=db,
            )
            summary = bundle.get("summary") or {}
            tier = str(bundle.get("tier") or "")
            prior_n = int(summary.get("n_visits") or prior_n)
            db.execute(
                """UPDATE fact_mo_case
                   SET history_prior_n=?, history_tier=?
                   WHERE mis_id=?""",
                (prior_n, tier, row["mis_id"]),
            )
            updated += 1
    db.commit()
    return {"refreshed": updated, "patients": len(by_key)}


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
    args = parser.parse_args()
    date.fromisoformat(args.date_from)
    date.fromisoformat(args.date_to)

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
    updated = 0
    if not args.dry_run:
        for visit_id, key in by_visit.items():
            cur = db.execute(
                """UPDATE fact_mo_case SET patient_key=?
                   WHERE visit_id=? AND TRIM(COALESCE(patient_key,''))=''""",
                (key, visit_id),
            )
            updated += cur.rowcount or 0
        for mid, key in by_mis.items():
            cur = db.execute(
                """UPDATE fact_mo_case SET patient_key=?
                   WHERE mis_id=? AND TRIM(COALESCE(patient_key,''))=''""",
                (key, mid),
            )
            updated += cur.rowcount or 0
        db.commit()
    keyed = db.execute(
        "SELECT COUNT(*) FROM fact_mo_case WHERE TRIM(COALESCE(patient_key,''))!=''"
    ).fetchone()[0]
    print("warehouse_updated", updated, "keyed_now", keyed, flush=True)
    if not args.dry_run and not args.skip_history_refresh:
        stats = _refresh_history(db)
        print("history_refresh", stats, flush=True)
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
