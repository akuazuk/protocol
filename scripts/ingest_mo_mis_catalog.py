#!/usr/bin/env python3
"""GCE: mis_data -> fact_mis_catalog (без result, без ФИО пациента).

  source /opt/protocol/deploy/gcp-app/load_mis_env.sh
  PYTHONPATH=/opt/protocol /opt/protocol/venv-mis/bin/python \
    scripts/ingest_mo_mis_catalog.py --from 2026-08-20 --to 2026-09-21
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import types
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _allow_ck_without_pydantic() -> None:
    """venv-mis has PyMySQL, not pydantic; skip clinical_knowledge/__init__.py."""
    try:
        import pydantic  # noqa: F401
        return
    except ImportError:
        pass
    pkg = types.ModuleType("clinical_knowledge")
    pkg.__path__ = [str(ROOT / "clinical_knowledge")]  # type: ignore[attr-defined]
    sys.modules["clinical_knowledge"] = pkg


_allow_ck_without_pydantic()
from clinical_knowledge.mo_mis_catalog import (  # noqa: E402
    DX_SHORT_MAX,
    ensure_lab_indexes,
    ensure_schema,
    upsert_catalog_rows,
)

# Только колонки mis_data. Не выбирать result / mis_protocol.
SELECT_SQL = """
SELECT visit_id, vdate, specialist_id, specialist_name, specialization,
       diagnos, filial, patient_id
FROM mis_data
WHERE vdate >= %s AND vdate < %s
"""


def _chunks(d0: date, d1: date) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    cur = d0
    while cur < d1:
        nxt = date(cur.year + (1 if cur.month == 12 else 0), 1 if cur.month == 12 else cur.month + 1, 1)
        if nxt > d1:
            nxt = d1
        if nxt <= cur:
            break
        out.append((cur, nxt))
        cur = nxt
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument(
        "--warehouse",
        default=os.environ.get("MO_WAREHOUSE")
        or "/var/data/medical_exams/warehouse/mo_analytics.sqlite",
    )
    ap.add_argument(
        "--lab-db",
        default=os.environ.get("MO_LAB_DB")
        or "/var/data/medical_exams/warehouse/mo_lab.sqlite",
    )
    args = ap.parse_args()
    d0 = date.fromisoformat(args.date_from)
    d1 = date.fromisoformat(args.date_to)
    if d1 <= d0:
        raise SystemExit("date_to must be after date_from")
    if "result" in SELECT_SQL.lower().split():
        raise SystemExit("catalog SQL must not touch result")

    import pymysql

    pw = (os.environ.get("KRAVIRA_DB_PASSWORD") or "").strip()
    if not pw:
        raise SystemExit("KRAVIRA_DB_PASSWORD missing; run via load_mis_env.sh on GCE")
    warehouse = Path(args.warehouse)
    warehouse.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(warehouse)
    ensure_schema(db)
    n_rows = 0
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
        for start, end in _chunks(d0, d1):
            cur.execute(SELECT_SQL, (start.isoformat(), end.isoformat()))
            batch = []
            for row in cur.fetchall():
                batch.append(
                    {
                        "visit_id": row[0],
                        "visit_date": str(row[1] or "")[:10],
                        "specialist_id": row[2],
                        "doctor_fio": row[3],
                        "specialization": row[4],
                        "dx_short": str(row[5] or "")[:DX_SHORT_MAX],
                        "filial": row[6],
                        "patient_id": row[7],
                    }
                )
                if len(batch) >= 500:
                    n_rows += upsert_catalog_rows(db, batch)
                    batch = []
            if batch:
                n_rows += upsert_catalog_rows(db, batch)
    finally:
        con.close()
        db.close()
    lab = Path(args.lab_db)
    if lab.is_file():
        lab_conn = sqlite3.connect(lab)
        ensure_lab_indexes(lab_conn)
        lab_conn.close()
    report = {
        "ok": True,
        "engine": "ingest_mo_mis_catalog_v1",
        "from": d0.isoformat(),
        "to": d1.isoformat(),
        "upserted": n_rows,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
