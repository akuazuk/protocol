"""Каталог МИС: поиск без result, покрытие, очередь ingest."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from clinical_knowledge.mo_mis_catalog import (
    coverage_payload,
    enqueue_ingest_job,
    process_next_ingest_job,
    search_labs,
    search_visits,
    upsert_catalog_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def _warehouse(tmp_path: Path) -> Path:
    path = tmp_path / "warehouse" / "mo_analytics.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _cases(path: Path, rows: list[tuple[str, str, str]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE fact_mo_case (
          mis_id TEXT PRIMARY KEY, visit_id TEXT, visit_date TEXT NOT NULL,
          specialty TEXT, filial TEXT, patient_key TEXT, diagnosis_text TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO fact_mo_case VALUES (?, ?, ?, '', '', '', ?)",
        [(f"c{vid}", vid, day, dx) for vid, day, dx in rows],
    )
    conn.commit()
    conn.close()


def test_search_visits_badges_and_typed_id(tmp_path: Path) -> None:
    warehouse = _warehouse(tmp_path)
    _cases(warehouse, [("10001", "2026-09-18", "отит")])
    conn = sqlite3.connect(warehouse)
    upsert_catalog_rows(
        conn,
        [
            {
                "visit_id": "10002",
                "visit_date": "2026-09-18",
                "doctor_fio": "Иванов",
                "specialization": "лор",
                "dx_short": "гайморит",
                "patient_id": "9",
            }
        ],
    )
    conn.close()
    scored = search_visits(q="отит", date_from="2026-09-01", date_to="2026-09-30", warehouse=warehouse)
    assert scored["ok"] is True
    assert scored["items"][0]["badge"] == "В аналитике"
    assert scored["items"][0]["case_id"] == "c10001"
    assert scored["coverage"]["found"] >= 2
    assert scored["coverage"]["in_analytics"] == 1

    missing = search_visits(q="10099", warehouse=warehouse, live_lookup=lambda _vid: None)
    assert missing["items"][0]["visit_id"] == "10099"
    assert missing["items"][0]["badge"] == "Не разобрано"
    assert "visit_id" in missing["empty_reason"]


def test_patient_spark_uses_analytics_flag(tmp_path: Path) -> None:
    warehouse = _warehouse(tmp_path)
    conn = sqlite3.connect(warehouse)
    conn.execute(
        """
        CREATE TABLE fact_mo_case (
          mis_id TEXT PRIMARY KEY, visit_id TEXT, visit_date TEXT NOT NULL,
          specialty TEXT, filial TEXT, patient_key TEXT, diagnosis_text TEXT
        )
        """
    )
    conn.execute("INSERT INTO fact_mo_case VALUES ('c1','30001','2026-09-18','','','pk1','отит')")
    conn.commit()
    conn.close()
    payload = search_visits(q="отит", date_from="2026-09-01", date_to="2026-09-30", warehouse=warehouse)
    spark = payload["items"][0]["patient_spark"]
    assert spark[0]["visit_id"] == "30001"
    assert spark[0]["in_analytics"] is True


def test_search_does_not_touch_result() -> None:
    source = (ROOT / "clinical_knowledge/mo_mis_catalog.py").read_text(encoding="utf-8")
    ingest = (ROOT / "scripts/ingest_mo_mis_catalog.py").read_text(encoding="utf-8")
    queue = (ROOT / "scripts/run_mo_ingest_queue.py").read_text(encoding="utf-8")
    assert "FROM mis_protocol" not in source
    assert "FROM mis_protocol" not in ingest
    block = ingest.split("SELECT_SQL =")[1].split('"""', 2)[1]
    assert "result" not in block.lower().split()
    assert "_allow_ck_without_pydantic" in ingest
    assert "_allow_ck_without_pydantic" in queue


def test_ingest_queue_and_process(tmp_path: Path) -> None:
    warehouse = _warehouse(tmp_path)
    seen: list[str] = []

    def fake_ingest(visit_id: str, *, data_root=None):
        seen.append(visit_id)
        conn = sqlite3.connect(warehouse)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fact_mo_case (
              mis_id TEXT PRIMARY KEY, visit_id TEXT, visit_date TEXT NOT NULL,
              specialty TEXT, filial TEXT, patient_key TEXT, diagnosis_text TEXT
            )
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO fact_mo_case VALUES (?, ?, '2026-09-18', '', '', '', '')",
            (f"c{visit_id}", visit_id),
        )
        conn.commit()
        conn.close()

    job = enqueue_ingest_job(visit_id="20001", warehouse=warehouse)
    assert job["status"] == "queued"
    done = process_next_ingest_job(warehouse=warehouse, ingest_fn=fake_ingest)
    assert done is not None
    assert done["status"] == "done"
    assert done["case_id"] == "c20001"
    assert seen == ["20001"]
    again = enqueue_ingest_job(visit_id="20001", warehouse=warehouse)
    assert again["already_in_analytics"] is True
    assert again["case_id"] == "c20001"


def test_search_labs_timeline(tmp_path: Path) -> None:
    lab = tmp_path / "mo_lab.sqlite"
    conn = sqlite3.connect(lab)
    conn.execute(
        """
        CREATE TABLE fact_mo_lab (
          indicator_name TEXT, type_name TEXT, test_date TEXT,
          value TEXT, unit TEXT, patient_key TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO fact_mo_lab VALUES (?, 'кровь', ?, ?, 'г/л', 'p1')",
        [("гемоглобин", "2026-09-01", "120"), ("гемоглобин", "2026-09-10", "128")],
    )
    conn.commit()
    conn.close()
    payload = search_labs(q="гемоглобин", date_from="2026-09-01", date_to="2026-09-30", lab_db=lab)
    assert payload["items"][0]["n"] == 2
    assert payload["timeline"][0]["numeric"] == 120.0
    assert payload["reference_available"] is False


def test_coverage_payload_empty(tmp_path: Path) -> None:
    warehouse = _warehouse(tmp_path)
    out = coverage_payload(date_from="2026-09-01", date_to="2026-09-30", warehouse=warehouse)
    assert out["ok"] is True
    assert out["found"] == 0
    assert "не KPI" in out["label_ru"]
