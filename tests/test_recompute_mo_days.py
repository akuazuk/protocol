"""recompute_mo_days resolves Render CSV without pandas."""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from scripts.recompute_mo_days import recompute_day, resolve_partition


def test_resolve_partition_prefers_secure_csv(tmp_path: Path) -> None:
    day = date(2026, 8, 1)
    secure = tmp_path / "secure_cases" / "2026" / "08"
    secure.mkdir(parents=True)
    csv_path = secure / "mo_2026-08-01.csv"
    csv_path.write_text("visit_id,date\n1,2026-08-01\n", encoding="utf-8")
    raw = tmp_path / "raw" / "2026" / "08"
    raw.mkdir(parents=True)
    (raw / "mo_2026-08-01.parquet").write_bytes(b"not-used")
    assert resolve_partition(tmp_path, day) == csv_path


def test_recompute_day_updates_llm_pending_from_grades(tmp_path: Path) -> None:
    day = date(2026, 8, 1)
    secure = tmp_path / "secure_cases" / "2026" / "08"
    secure.mkdir(parents=True)
    reports = tmp_path / "reports" / "2026" / "08" / "01"
    reports.mkdir(parents=True)
    warehouse = tmp_path / "warehouse" / "mo_analytics.sqlite"
    warehouse.parent.mkdir(parents=True)

    with (secure / "mo_2026-08-01.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "visit_id",
                "mis_id",
                "date",
                "doctor_fio",
                "doctor_specialization",
                "filial",
                "kz_kind",
                "complaints",
                "diagnosis_clinical",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "visit_id": "100",
                "mis_id": "m1",
                "date": "2026-08-01",
                "doctor_fio": "Иванов И.И.",
                "doctor_specialization": "терапевт",
                "filial": "ул. Захарова, 50Д",
                "kz_kind": "kz",
                "complaints": "боль",
                "diagnosis_clinical": "J06",
            }
        )

    case = {
        "visit_id": "100",
        "mis_id": "m1",
        "date": "2026-08-01",
        "overall_pct": 70.0,
        "status": "review",
        "doctor_fio": "Иванов И.И.",
        "doctor_specialization": "терапевт",
        "filial": "ул. Захарова, 50Д",
        "deep": {"findings": [], "axes": {}, "n_by_severity": {}},
    }
    (secure / "kz_l1_2026-08-01_cases.jsonl").write_text(
        json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (secure / "kz_l1_2026-08-01_llm_queue.json").write_text(
        json.dumps({"visit_ids": ["100"], "n": 1}), encoding="utf-8"
    )
    (secure / "kz_l1_2026-08-01_llm_grades.jsonl").write_text(
        json.dumps({"visit_id": "100", "overall_pct": 40, "verdict": "weak"}) + "\n",
        encoding="utf-8",
    )
    (reports / "report.json").write_text(
        json.dumps(
            {
                "run_id": "t1",
                "revision": 1,
                "summary": {"avg_score": 70.0},
                "completeness": {"llm_queue_pending": 1},
            }
        ),
        encoding="utf-8",
    )

    result = recompute_day(day, data_root=tmp_path, warehouse=warehouse, write_reports=True)
    assert result["status"] == "success"
    assert result["llm_queue_pending"] == 0
    assert result["was_llm_queue_pending"] == 1
    updated = json.loads((reports / "report.json").read_text(encoding="utf-8"))
    assert updated["completeness"]["llm_queue_pending"] == 0
