#!/usr/bin/env python3
"""Создать индексы склада МО (идемпотентно). Только GCE / writable sqlite.

  sudo python3 scripts/ensure_mo_warehouse_indexes.py
  MO_ANALYTICS_DB=/var/data/medical_exams/warehouse/mo_analytics.sqlite \\
    python3 scripts/ensure_mo_warehouse_indexes.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_case_date ON fact_mo_case(visit_date)",
    "CREATE INDEX IF NOT EXISTS idx_case_diagnosis_text ON fact_mo_case(diagnosis_text)",
    "CREATE INDEX IF NOT EXISTS idx_fact_mo_case_visit ON fact_mo_case(visit_id)",
    "CREATE INDEX IF NOT EXISTS idx_case_zone_bands ON fact_mo_case(visit_date, zone1_band, zone2a_band, zone2b_band)",
    "CREATE INDEX IF NOT EXISTS idx_case_attention_date ON fact_mo_case(attention_primary, visit_date)",
    "CREATE INDEX IF NOT EXISTS idx_finding_mis ON fact_mo_finding(mis_id)",
    "CREATE INDEX IF NOT EXISTS idx_finding_code ON fact_mo_finding(finding_code, severity)",
)


def db_path() -> Path:
    configured = (os.environ.get("MO_ANALYTICS_DB") or "").strip()
    if configured:
        return Path(configured)
    return Path("/var/data/medical_exams/warehouse/mo_analytics.sqlite")


def main() -> int:
    path = db_path()
    if not path.is_file():
        print(f"missing {path}", file=sys.stderr)
        return 2
    conn = sqlite3.connect(str(path), timeout=30)
    try:
        for stmt in INDEX_SQL:
            conn.execute(stmt)
        conn.commit()
        names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%' ORDER BY 1"
            )
        ]
    finally:
        conn.close()
    print("ok", path)
    print("indexes", ",".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
