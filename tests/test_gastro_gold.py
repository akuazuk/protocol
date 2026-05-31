"""Gold-set consult cases for gastro MVP rule checker."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge import extract_consult_facts_heuristic, run_rule_checker

GOLD = Path(__file__).resolve().parent.parent / "data" / "gastro_mvp" / "consult_gold.jsonl"


def _load_gold() -> list[dict]:
    if not GOLD.is_file():
        return []
    out: list[dict] = []
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def test_gold_gerd_full_diagnosis_passes():
    for row in _load_gold():
        if row.get("consultation_id") != "gold_gerd_full_01":
            continue
        facts = extract_consult_facts_heuristic(
            row["text"], demographics_meta=row.get("patient_context")
        )
        res = run_rule_checker(facts, condition_ids=["gerd"])
        formula = next(f for f in res["findings"] if f.get("rule_type") == "diagnosis_formula")
        assert formula.get("passed") is True
        return
    raise AssertionError("gold case not found")


def test_gold_gerd_incomplete_fails_formula():
    for row in _load_gold():
        if row.get("consultation_id") != "gold_gerd_incomplete_01":
            continue
        facts = extract_consult_facts_heuristic(
            row["text"], demographics_meta=row.get("patient_context")
        )
        res = run_rule_checker(facts, condition_ids=["gerd"])
        formula = next(f for f in res["findings"] if f.get("rule_type") == "diagnosis_formula")
        assert formula.get("passed") is False
        return
    raise AssertionError("gold case not found")


def test_gold_child_population_critical():
    for row in _load_gold():
        if row.get("consultation_id") != "gold_gerd_child_mismatch_01":
            continue
        facts = extract_consult_facts_heuristic(
            row["text"], demographics_meta=row.get("patient_context")
        )
        res = run_rule_checker(facts, condition_ids=["gerd"])
        crit = [f for f in res["findings"] if f.get("severity") == "critical" and not f.get("passed")]
        assert crit
        return
    raise AssertionError("gold case not found")
