import sqlite3
from datetime import date

from clinical_knowledge.mo_daily import initialize_warehouse
from scripts.backfill_mo_warehouse import _months, prune_after


def test_month_range_is_inclusive() -> None:
    assert _months("2026-01", "2026-03") == ["2026-01", "2026-02", "2026-03"]


def test_prune_after_removes_only_future_local_facts(tmp_path) -> None:
    warehouse = tmp_path / "warehouse.sqlite"
    initialize_warehouse(warehouse)
    with sqlite3.connect(warehouse) as db:
        db.execute(
            "INSERT INTO fact_mo_case VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("1", "11", "2026-07-27", "medical_exam", 80, "good", "", "", "", "a", "now"),
        )
        db.execute(
            "INSERT INTO fact_mo_case VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("2", "22", "2026-07-28", "medical_exam", 70, "review", "", "", "", "b", "now"),
        )
        db.execute(
            "INSERT INTO fact_mo_score_axis VALUES (?, ?, ?)",
            ("2", "documentation", 70),
        )
        db.commit()
    assert prune_after(warehouse, date(2026, 7, 27)) == 1
    with sqlite3.connect(warehouse) as db:
        assert db.execute("SELECT mis_id FROM fact_mo_case").fetchall() == [("1",)]
        assert db.execute("SELECT count(*) FROM fact_mo_score_axis").fetchone()[0] == 0
