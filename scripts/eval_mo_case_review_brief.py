#!/usr/bin/env python3
"""Eval coverage clinical gaps + review_brief на gold fixture(s).

Пример:
  python3 scripts/eval_mo_case_review_brief.py
  python3 scripts/eval_mo_case_review_brief.py --fixture tests/fixtures/mo_case_review_brief_mo1_gold.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.mo_case_review_brief import build_case_review_brief
from clinical_knowledge.mo_clinical_gaps import merge_clinical_gaps_into_findings


def _eval_one(gold: dict) -> dict:
    clinical = gold.get("clinical") or {}
    expect_codes = list(gold.get("expect_gap_codes") or [])
    findings = merge_clinical_gaps_into_findings([], clinical)
    got = {str(f.get("code") or "") for f in findings}
    hit = [c for c in expect_codes if c in got]
    miss = [c for c in expect_codes if c not in got]
    detail = {
        "zones": gold.get("zones_stub") or {"ok": False},
        "findings": findings,
        "icd_visit_status": gold.get("icd_visit_status") or {"status": "ok"},
        "protocol_suggest": gold.get("protocol_suggest") or {"available": False},
        "patient_history": gold.get("patient_history") or {"summary": {"n_visits": 0}},
        "record": gold.get("record")
        or {
            "visit_id": gold.get("id"),
            "clinical_diagnosis": clinical.get("clinical_diagnosis"),
        },
    }
    brief = build_case_review_brief(detail)
    expect_brief = gold.get("expect_brief") or {}
    feedback = brief.get("doctor_feedback") or []
    blob = " ".join(str(x) for x in feedback).lower()
    substr_ok = all(s.lower() in blob for s in (expect_brief.get("must_mention_substrings") or []))
    return {
        "id": gold.get("id"),
        "gaps_expected": len(expect_codes),
        "gaps_hit": len(hit),
        "gaps_miss": miss,
        "gap_recall": (len(hit) / len(expect_codes)) if expect_codes else None,
        "brief_ok": bool(brief.get("ok") and brief.get("available")),
        "feedback_n": len(feedback),
        "feedback_min_ok": len(feedback) >= int(expect_brief.get("min_doctor_feedback") or 0),
        "feedback_substr_ok": substr_ok,
        "diagnosis_axes": (brief.get("diagnosis_axes") or {}),
        "summary_ru": brief.get("summary_ru"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fixture",
        type=Path,
        action="append",
        default=None,
        help="Gold JSON (можно несколько). Default: mo1 fixture.",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    fixtures = args.fixture or [
        ROOT / "tests" / "fixtures" / "mo_case_review_brief_mo1_gold.json",
    ]
    rows = []
    for path in fixtures:
        gold = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_eval_one(gold))
    recalls = [r["gap_recall"] for r in rows if r["gap_recall"] is not None]
    report = {
        "n": len(rows),
        "avg_gap_recall": (sum(recalls) / len(recalls)) if recalls else None,
        "all_brief_ok": all(r["brief_ok"] for r in rows),
        "all_feedback_ok": all(r["feedback_min_ok"] and r["feedback_substr_ok"] for r in rows),
        "cases": rows,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    ok = report["all_brief_ok"] and report["all_feedback_ok"]
    if recalls and min(recalls) < 0.7:
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
