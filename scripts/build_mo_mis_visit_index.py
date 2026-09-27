#!/usr/bin/env python3
"""P1: лёгкий индекс всех клиентов МИС в паспорт (без полного result).

Канон: GCE, пароль из Secret Manager / .env.mis. В stdout только счётчики.
Оценку визитов не запускает. Не импортирует clinical_knowledge на старте.

  source /opt/protocol/deploy/gcp-app/load_mis_env.sh
  PYTHONPATH=/opt/protocol python3 /opt/protocol/scripts/build_mo_mis_visit_index.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RE_MKB = re.compile(r"\b([A-TV-Z][0-9]{2}(?:\.[0-9]{1,2})?)\b")
DX_LABEL_MAX = 80


def patient_key_for(patient_id: object) -> str:
    normalized = str(patient_id or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20] if normalized else ""


def _load_passport():
    path = ROOT / "clinical_knowledge" / "mo_patient_passport.py"
    spec = importlib.util.spec_from_file_location("mo_patient_passport_p1", path)
    if spec is None or spec.loader is None:
        raise SystemExit("passport_module_missing")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dx_label(value: object) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= DX_LABEL_MAX:
        return text
    return text[: DX_LABEL_MAX - 1].rstrip() + "..."


def _mkb(value: object) -> str:
    found = _RE_MKB.search(str(value or ""))
    return found.group(1).upper() if found else ""


def _month_starts(d0: date, d1: date) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    cur = date(d0.year, d0.month, 1)
    if d0.day != 1:
        cur = d0
    while cur < d1:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        out.append((cur, min(nxt, d1)))
        cur = min(nxt, d1)
    return out


def _connect_mis():
    import pymysql

    pw = (os.environ.get("KRAVIRA_DB_PASSWORD") or "").strip()
    if not pw:
        raise SystemExit("no_mis_password")
    return pymysql.connect(
        host=(os.environ.get("KRAVIRA_DB_HOST") or "178.163.240.131").strip(),
        port=int(os.environ.get("KRAVIRA_DB_PORT") or "6330"),
        user=(os.environ.get("KRAVIRA_DB_USER") or "kravira_mc_user").strip(),
        password=pw,
        database=(os.environ.get("KRAVIRA_DB_NAME") or "kravira_mc").strip(),
        charset="utf8mb4",
        connect_timeout=30,
        read_timeout=600,
    )


def _iso(value: object) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()[:10]
    return str(value or "")[:10]


def _table_cols(cur, table: str) -> set[str]:
    cur.execute(f"SHOW COLUMNS FROM `{table}`")
    return {str(row[0]).lower() for row in cur.fetchall()}


def _discover_range(cur) -> tuple[str, str]:
    cur.execute(
        """
        SELECT MIN(date), MAX(date)
          FROM mis_protocol
         WHERE patient_id IS NOT NULL AND CAST(patient_id AS CHAR) <> ''
        """
    )
    lo, hi = cur.fetchone()
    start = _iso(lo) or "2018-01-01"
    end = _iso(hi)
    if not end:
        raise SystemExit("mis_protocol_empty")
    # полуинтервал: день после max
    y, m, d = [int(x) for x in end.split("-")]
    nxt = date(y, m, d).toordinal() + 1
    return start, date.fromordinal(nxt).isoformat()


def _fetch_protocol_month(cur, start: str, end: str) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT id, date, visit_id, patient_id
          FROM mis_protocol
         WHERE date >= %s AND date < %s
           AND patient_id IS NOT NULL AND CAST(patient_id AS CHAR) <> ''
           AND visit_id IS NOT NULL AND CAST(visit_id AS CHAR) <> ''
        """,
        (start, end),
    )
    out: list[dict[str, Any]] = []
    for mid, day, visit_id, patient_id in cur.fetchall():
        key = patient_key_for(patient_id)
        if not key:
            continue
        out.append(
            {
                "patient_key": key,
                "visit_id": str(visit_id).strip(),
                "visit_date": _iso(day),
                "mis_id": str(mid),
                "document_kind": "mis_index",
            }
        )
    return out


def _fetch_mis_data_month(cur, start: str, end: str) -> dict[str, dict[str, str]]:
    cols = _table_cols(cur, "mis_data")
    date_col = next((c for c in ("vdate", "date", "visit_date") if c in cols), "")
    spec_col = "specialization" if "specialization" in cols else ""
    dx_col = next((c for c in ("mis_diagnos", "diagnos", "diagnosis") if c in cols), "")
    if "visit_id" not in cols or not date_col:
        return {}
    select = [f"`{date_col}`", "visit_id"]
    if spec_col:
        select.append(f"`{spec_col}`")
    if dx_col:
        select.append(f"`{dx_col}`")
    cur.execute(
        f"""
        SELECT {", ".join(select)}
          FROM mis_data
         WHERE `{date_col}` >= %s AND `{date_col}` < %s
           AND visit_id IS NOT NULL
        """,
        (start, end),
    )
    out: dict[str, dict[str, str]] = {}
    for row in cur.fetchall():
        visit_id = str(row[1] or "").strip()
        if not visit_id:
            continue
        rec = out.setdefault(visit_id, {"specialty": "", "diagnosis_code": "", "dx_label": ""})
        idx = 2
        if spec_col:
            spec = str(row[idx] or "").strip()
            idx += 1
            if spec and not rec["specialty"]:
                rec["specialty"] = spec[:80]
        if dx_col:
            dx = str(row[idx] or "").strip()
            if dx:
                code = _mkb(dx)
                if code and not rec["diagnosis_code"]:
                    rec["diagnosis_code"] = code
                if not rec["dx_label"]:
                    rec["dx_label"] = _dx_label(dx)
    return out


def _fetch_lab_coverage(cur) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT patient_id,
               COUNT(*) AS n_rows,
               COUNT(DISTINCT date) AS n_dates,
               MIN(date) AS first_date,
               MAX(date) AS last_date
          FROM mis_tests
         WHERE patient_id IS NOT NULL AND CAST(patient_id AS CHAR) <> ''
         GROUP BY patient_id
        """
    )
    out: list[dict[str, Any]] = []
    for patient_id, n_rows, n_dates, first_date, last_date in cur.fetchall():
        key = patient_key_for(patient_id)
        if not key:
            continue
        out.append(
            {
                "patient_key": key,
                "n_lab_rows": int(n_rows or 0),
                "n_lab_dates": int(n_dates or 0),
                "first_date": _iso(first_date),
                "last_date": _iso(last_date),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", default="")
    parser.add_argument("--to", dest="date_to", default="")
    parser.add_argument(
        "--warehouse",
        default=os.environ.get("MO_WAREHOUSE")
        or "/var/data/medical_exams/warehouse/mo_analytics.sqlite",
    )
    parser.add_argument("--lab", default="")
    parser.add_argument("--skip-labs", action="store_true")
    parser.add_argument("--skip-mis-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    warehouse = Path(args.warehouse)
    if not warehouse.is_file():
        print(json.dumps({"ok": False, "error": "warehouse_missing"}, ensure_ascii=False))
        return 2
    passport = _load_passport()
    lab_path = Path(args.lab) if args.lab else passport.default_lab_path(warehouse)
    con = _connect_mis()
    try:
        cur = con.cursor()
        date_from = args.date_from
        date_to = args.date_to
        if not date_from or not date_to:
            discovered = _discover_range(cur)
            date_from = date_from or discovered[0]
            date_to = date_to or discovered[1]
        date.fromisoformat(date_from)
        date.fromisoformat(date_to)
        chunks = _month_starts(date.fromisoformat(date_from), date.fromisoformat(date_to))
        print(
            json.dumps(
                {
                    "phase": "range",
                    "from": date_from,
                    "to": date_to,
                    "months": len(chunks),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        inserted = 0
        merged = 0
        protocol_rows = 0
        unique_keys: set[str] = set()
        for start, end in chunks:
            cards = _fetch_protocol_month(cur, start.isoformat(), end.isoformat())
            protocol_rows += len(cards)
            if not args.skip_mis_data:
                extra = _fetch_mis_data_month(cur, start.isoformat(), end.isoformat())
                for card in cards:
                    meta = extra.get(card["visit_id"]) or {}
                    card["specialty"] = meta.get("specialty") or ""
                    card["diagnosis_code"] = meta.get("diagnosis_code") or ""
                    card["dx_label"] = meta.get("dx_label") or ""
            unique_keys.update(c["patient_key"] for c in cards)
            if args.dry_run:
                merged += len(cards)
                print(
                    json.dumps(
                        {
                            "phase": "month",
                            "from": start.isoformat(),
                            "to": end.isoformat(),
                            "rows": len(cards),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            stats = passport.merge_mis_visit_cards(
                warehouse,
                cards,
                lab_path=lab_path,
                refresh=False,
            )
            inserted += int(stats.get("inserted_cards") or 0)
            merged += int(stats.get("merged_cards") or 0)
            print(
                json.dumps(
                    {
                        "phase": "month",
                        "from": start.isoformat(),
                        "to": end.isoformat(),
                        "rows": len(cards),
                        "inserted": stats.get("inserted_cards"),
                        "visit_cards": stats.get("visit_cards"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        lab_stats: dict[str, Any] = {"skipped": True}
        if not args.skip_labs:
            coverage = _fetch_lab_coverage(cur)
            unique_keys.update(r["patient_key"] for r in coverage)
            if args.dry_run:
                lab_stats = {"dry_run": True, "coverage_fetched": len(coverage)}
            else:
                lab_stats = passport.merge_mis_lab_coverage(
                    warehouse,
                    coverage,
                    lab_path=lab_path,
                    refresh=False,
                )
        refresh = {"skipped": True}
        if not args.dry_run:
            refresh = passport.refresh_passport_aggregates(warehouse, lab_path=lab_path)
        report = {
            "ok": True,
            "dry_run": bool(args.dry_run),
            "from": date_from,
            "to": date_to,
            "protocol_rows": protocol_rows,
            "unique_keys_seen": len(unique_keys),
            "inserted_cards": inserted,
            "merged_cards": merged,
            "lab": {
                k: lab_stats.get(k)
                for k in (
                    "ok",
                    "coverage_rows",
                    "lab_passports",
                    "merged",
                    "skipped",
                    "dry_run",
                    "coverage_fetched",
                )
                if k in lab_stats
            },
            "refresh": {
                k: refresh.get(k)
                for k in (
                    "ok",
                    "passports",
                    "visit_cards",
                    "specialty_rows",
                    "lab_passports",
                    "skipped",
                )
                if k in refresh
            },
        }
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
