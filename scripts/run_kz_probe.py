#!/usr/bin/env python3
"""Прогон probe-кейсов подбора КП (без полного RAG)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.consult_retrieval import consult_target_protocol_paths


def _facts(case: dict) -> dict:
    return {
        "consultation": {
            "complaints": case.get("complaints") or [],
            "diagnosis_text": " ".join(case.get("icd") or []),
            "conditions_hint": case.get("complaints") or [],
            "performed_exams": [],
        },
        "patient_context": {"adult_or_child": "adult"},
    }


def run_probe(cases_path: Path, *, min_score: float = 22.0) -> list[dict]:
    results: list[dict] = []
    with cases_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            paths, meta = consult_target_protocol_paths(
                diag_icd=case.get("icd"),
                merged_icd=case.get("icd"),
                clinical_rules=None,
                specialty_slugs=[case["specialty_slug"]] if case.get("specialty_slug") else [],
                consult_facts=_facts(case),
                primary_specialty=case.get("specialty_slug"),
                min_match_score=min_score,
            )
            matches = meta.get("protocol_matches") or []
            rejected = meta.get("rejected_protocols") or []
            top = matches[0] if matches else {}
            results.append({
                "id": case.get("id"),
                "paths_count": len(paths),
                "top_title": (top.get("title") or "")[:80],
                "top_score": top.get("match_score"),
                "top_flags": top.get("pick_risk_flags") or [],
                "rejected_admin": any(
                    "admin_order" in (r.get("pick_risk_flags") or []) for r in rejected
                ),
                "ok": bool(paths),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Пробный подбор КП для кейсов КЗ")
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data" / "ml" / "kz_probe_cases.jsonl",
    )
    parser.add_argument("--min-score", type=float, default=22.0)
    args = parser.parse_args()
    if not args.cases.exists():
        print(json.dumps({"error": f"not found: {args.cases}"}, ensure_ascii=False))
        sys.exit(1)
    out = run_probe(args.cases, min_score=args.min_score)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    failed = [r for r in out if not r.get("ok")]
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
