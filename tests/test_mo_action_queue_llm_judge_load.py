"""Action-queue LLM judge: load full queue from report (not truncated)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.run_mo_action_queue_llm_judge import load_action_items_local


def test_load_action_items_local_falls_back_to_action_queue(tmp_path: Path) -> None:
    day = "2026-08-06"
    y, m, d = day.split("-")
    report_dir = tmp_path / "reports" / y / m / d
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "action_cases": None,
                "action_queue": [
                    {"visit_id": "111", "mis_id": "m1", "reason": "P0"},
                    {"visit_id": "222", "mis_id": "m2", "reason": "P1"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    items = load_action_items_local(day, medical_root=tmp_path)
    assert len(items) == 2
    assert items[0]["visit_id"] == "111"


def test_load_action_items_local_empty_action_cases_items_uses_queue(tmp_path: Path) -> None:
    day = "2026-08-06"
    y, m, d = day.split("-")
    report_dir = tmp_path / "reports" / y / m / d
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "action_cases": {"available": False, "items": []},
                "action_queue": [{"visit_id": "333", "mis_id": "m3"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    items = load_action_items_local(day, medical_root=tmp_path)
    assert [it["visit_id"] for it in items] == ["333"]
