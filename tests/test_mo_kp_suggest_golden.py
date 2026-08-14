"""CI: верно/неверно находится КП по диагнозу МО (каталог protocol_cards)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_knowledge.mo_kp_suggest_golden_eval import (
    evaluate_mo_kp_suggest_row,
    load_mo_kp_suggest_golden,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "mo_kp_suggest_golden.jsonl"


def _has_visit_date(row: dict) -> bool:
    clinical = row.get("clinical") or {}
    record = row.get("record") or {}
    return bool(clinical.get("visit_date") or record.get("date") or record.get("visit_date"))


def _has_age_signal(row: dict) -> bool:
    clinical = row.get("clinical") or {}
    record = row.get("record") or {}
    keys = ("patient_age_years", "age_years", "patient_bdate", "birth_date")
    return any(clinical.get(k) not in (None, "") or record.get(k) not in (None, "") for k in keys)


def test_golden_fixture_loads() -> None:
    rows = load_mo_kp_suggest_golden(FIX)
    assert len(rows) >= 40
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert "flatfoot_child_ortho_positive" in ids
    assert "new_orvi_ignores_old_flatfoot_history" in ids
    assert "specialty_only_no_filler" in ids
    for row in rows:
        assert _has_visit_date(row), f"{row['id']}: visit_date required"
        if str(row["id"]).startswith("unknown_age"):
            continue
        assert _has_age_signal(row), f"{row['id']}: age_years or bdate required"
        blob = json.dumps(row, ensure_ascii=False).lower()
        assert "patient_id" not in blob
        assert "фио" not in blob


@pytest.mark.parametrize("row", load_mo_kp_suggest_golden(FIX), ids=lambda r: r["id"])
def test_mo_kp_suggest_golden_row(row: dict) -> None:
    result = evaluate_mo_kp_suggest_row(row)
    assert result["ok"], f"{row['id']}: {result['errors']} top={result.get('top_path')}"


def test_mo_kp_suggest_golden_suite_size() -> None:
    rows = load_mo_kp_suggest_golden(FIX)
    assert len(rows) >= 40
