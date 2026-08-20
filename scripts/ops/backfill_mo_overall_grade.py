#!/usr/bin/env python3
"""Пересчитать overall_grade на складе и убрать unmatched+bad план.

Без PHI в stdout. Запуск на GCE:

  docker exec protocol-web python3 /app/scripts/ops/backfill_mo_overall_grade.py

Окно по умолчанию: 2026-07-26 .. сегодня. Не делает deep-rescore.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import date

from clinical_knowledge.mo_overall_grade import compute_mo_overall_grade


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.environ.get("MO_WAREHOUSE_DB", ""))
    parser.add_argument("--from", dest="date_from", default="2026-07-26")
    parser.add_argument("--to", dest="date_to", default=date.today().isoformat())
    args = parser.parse_args()
    db_path = args.db or os.environ.get("MO_ANALYTICS_DB") or ""
    if not db_path:
        root = os.environ.get("MO_DATA_ROOT", "/var/data/medical_exams")
        db_path = os.path.join(root, "warehouse", "mo_analytics.sqlite")
    if not os.path.isfile(db_path):
        raise SystemExit(f"warehouse not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cols = {row[1] for row in conn.execute("PRAGMA table_info(fact_mo_case)")}
    for name, decl in (
        ("overall_grade", "TEXT"),
        ("overall_grade_ru", "TEXT"),
        ("overall_grade_reason_ru", "TEXT"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE fact_mo_case ADD COLUMN {name} {decl}")

    unmatched_bad = conn.execute(
        """UPDATE fact_mo_case
           SET zone2b_band='na'
           WHERE document_kind IN ('clinical_visit','consultation')
             AND visit_date BETWEEN ? AND ?
             AND COALESCE(zone2b_kp_status,'') != 'matched'
             AND zone2b_band='bad'""",
        (args.date_from, args.date_to),
    ).rowcount

    rows = conn.execute(
        """SELECT mis_id, zone1_band, zone2a_band, zone2b_band, zone2b_kp_status,
                  attention_primary
           FROM fact_mo_case
           WHERE document_kind IN ('clinical_visit','consultation')
             AND visit_date BETWEEN ? AND ?""",
        (args.date_from, args.date_to),
    ).fetchall()
    counts: dict[str, int] = {}
    for row in rows:
        grade = compute_mo_overall_grade(
            {
                "zone1_band": row["zone1_band"],
                "zone2a_band": row["zone2a_band"],
                "zone2b_band": row["zone2b_band"],
                "zone2b_kp_status": row["zone2b_kp_status"],
                "safety": {
                    "band": "important"
                    if str(row["attention_primary"] or "") == "safety"
                    else "none"
                },
            }
        )
        conn.execute(
            """UPDATE fact_mo_case
               SET overall_grade=?, overall_grade_ru=?, overall_grade_reason_ru=?
               WHERE mis_id=?""",
            (grade["grade"], grade["label_ru"], grade["reason_ru"], row["mis_id"]),
        )
        counts[grade["grade"]] = counts.get(grade["grade"], 0) + 1
    conn.commit()
    leftover = conn.execute(
        """SELECT COUNT(*) FROM fact_mo_case
           WHERE document_kind IN ('clinical_visit','consultation')
             AND visit_date BETWEEN ? AND ?
             AND COALESCE(zone2b_kp_status,'') != 'matched'
             AND zone2b_band='bad'""",
        (args.date_from, args.date_to),
    ).fetchone()[0]
    conn.close()
    print(
        f"ok window={args.date_from}..{args.date_to} n={len(rows)} "
        f"unmatched_bad_fixed={unmatched_bad} leftover_unmatched_bad={leftover} "
        f"grades={counts}"
    )
    return 0 if leftover == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
