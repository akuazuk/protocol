"""Проверка обезличенных fixtures eval/mo_concordance (E0/E2)."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.mo_concordance_findings import evaluate_mo_concordance

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "eval" / "mo_concordance"


def _load(name: str) -> list[dict]:
    path = FIX / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_eval_fixtures_exist_and_roundtrip() -> None:
    pos = _load("positives.jsonl")
    neg = _load("negatives.jsonl")
    assert len(pos) >= 5
    assert len(neg) >= 5
    for row in pos:
        case = row["case"]
        assert "patient_id" not in case and "visit_id" not in case
        assert "doctor_fio" not in case
        got = {f["code"] for f in evaluate_mo_concordance(case)}
        # expected_codes - снимок на момент калибровки; допускаем subset
        expected = set(row.get("expected_codes") or [])
        assert expected, "positive fixture must declare expected_codes"
        assert got & expected, f"got={got} expected_any_of={expected}"
    for row in neg:
        codes = {f["code"] for f in evaluate_mo_concordance(row["case"])}
        # negative: не должно быть P1 concordance
        assert "finding_not_in_diagnosis" not in codes
        assert "underworkup_chronic_red_flag" not in codes
