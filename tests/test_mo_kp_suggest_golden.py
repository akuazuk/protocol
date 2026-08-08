"""CI: верно/неверно находится КП по диагнозу МО (каталог protocol_cards)."""
from __future__ import annotations

from pathlib import Path

import pytest

from clinical_knowledge.mo_kp_suggest_golden_eval import (
    evaluate_mo_kp_suggest_golden,
    evaluate_mo_kp_suggest_row,
    load_mo_kp_suggest_golden,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "mo_kp_suggest_golden.jsonl"


def test_golden_fixture_loads() -> None:
    rows = load_mo_kp_suggest_golden(FIX)
    assert len(rows) >= 5
    ids = {r["id"] for r in rows}
    assert "flatfoot_child_ortho_positive" in ids
    assert "new_orvi_ignores_old_flatfoot_history" in ids


@pytest.mark.parametrize("row", load_mo_kp_suggest_golden(FIX), ids=lambda r: r["id"])
def test_mo_kp_suggest_golden_row(row: dict) -> None:
    result = evaluate_mo_kp_suggest_row(row)
    assert result["ok"], f"{row['id']}: {result['errors']} top={result.get('top_path')}"


def test_mo_kp_suggest_golden_suite_pass_rate() -> None:
    summary = evaluate_mo_kp_suggest_golden(FIX)
    assert summary["n"] >= 5
    assert summary["failed"] == 0, [
        (r["id"], r["errors"]) for r in summary["results"] if not r["ok"]
    ]
