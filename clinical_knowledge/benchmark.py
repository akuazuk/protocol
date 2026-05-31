"""Бенчмарк rule checker на gold-set consult_gold.jsonl."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .consult_facts import extract_consult_facts_heuristic
from .loader import load_conditions, load_rules_by_condition
from .rule_checker import run_rule_checker

GOLD_PATH = Path(__file__).resolve().parent.parent / "data" / "gastro_mvp" / "consult_gold.jsonl"
BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "data" / "gastro_mvp" / "benchmark.json"


def _load_gold_cases(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or GOLD_PATH
    if not p.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _eval_case(case: dict[str, Any]) -> dict[str, Any]:
    expect = case.get("expect") or {}
    cid = str(case.get("target_condition") or "")
    facts = extract_consult_facts_heuristic(
        str(case.get("text") or ""),
        demographics_meta=case.get("patient_context") or {},
    )
    result = run_rule_checker(facts, condition_ids=[cid] if cid else None)
    findings = result.get("findings") or []
    checks: dict[str, bool] = {}
    ok = True

    if "diagnosis_formula_pass" in expect:
        formula = next(
            (f for f in findings if f.get("rule_type") == "diagnosis_formula"),
            None,
        )
        passed = bool(formula and formula.get("passed"))
        checks["diagnosis_formula_pass"] = passed == bool(expect["diagnosis_formula_pass"])
        ok = ok and checks["diagnosis_formula_pass"]

    if expect.get("population_mismatch"):
        crit = any(
            f.get("severity") == "critical" and not f.get("passed") for f in findings
        )
        checks["population_mismatch"] = crit
        ok = ok and checks["population_mismatch"]

    if expect.get("has_condition_hint"):
        hint = expect["has_condition_hint"]
        hints = facts.get("consultation", {}).get("conditions_hint") or []
        checks["has_condition_hint"] = hint in hints
        ok = ok and checks["has_condition_hint"]

    return {
        "consultation_id": case.get("consultation_id"),
        "target_condition": cid,
        "ok": ok,
        "checks": checks,
        "rules_compliance_pct": result.get("rules_compliance_pct"),
    }


def run_gastro_gold_benchmark(gold_path: Path | None = None) -> dict[str, Any]:
    cases = _load_gold_cases(gold_path)
    rows = [_eval_case(c) for c in cases]
    passed = sum(1 for r in rows if r.get("ok"))
    total = len(rows)
    return {
        "title": "Эталон проверки КЗ по правилам (гастро MVP)",
        "scope": "data/gastro_mvp/consult_gold.jsonl",
        "cases_total": total,
        "cases_passed": passed,
        "pass_rate_pct": round(100.0 * passed / total, 1) if total else 0,
        "conditions_loaded": len(load_conditions()),
        "rules_loaded": sum(len(v) for v in load_rules_by_condition().values()),
        "updated": date.today().isoformat(),
        "cases": rows,
        "methodology_ru": (
            "Детерминированный rule_checker на размеченных синтетических КЗ; "
            "пересчёт: python3 scripts/update_gastro_rules_benchmark.py"
        ),
    }


def write_gastro_benchmark(out_path: Path | None = None) -> dict[str, Any]:
    payload = run_gastro_gold_benchmark()
    path = out_path or BENCHMARK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
