#!/usr/bin/env python3
"""Сравнение подбора КП: карты как есть vs overlay паспортов Ilex. Без PHI."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NOSOLOGY_CASES: list[dict[str, Any]] = [
    {
        "id": "hypertension_i11",
        "clinical": {
            "clinical_diagnosis": "Гипертоническая болезнь",
            "mis_diagnos": "I11.9",
            "patient_age_years": 61,
            "visit_date": "2026-07-15",
        },
        "record": {"visit_id": "eval_i11", "specialty": "Терапевт", "date": "2026-07-15"},
        "expect_any": ["гипертенз", "гипертон", "давлен"],
        "reject_any": ["стволов", "гемопоэтическ", "экстренн", "дискинез"],
    },
    {
        "id": "bronchitis_j20",
        "clinical": {
            "clinical_diagnosis": "Острый бронхит",
            "mis_diagnos": "J20.9",
            "patient_age_years": 44,
            "visit_date": "2026-07-15",
        },
        "record": {"visit_id": "eval_j20", "specialty": "Терапевт", "date": "2026-07-15"},
        "expect_any": ["бронхит"],
        "reject_any": ["стволов", "гемопоэтическ", "экстренн", "дискинез"],
    },
    {
        "id": "sinusitis_j32",
        "clinical": {
            "clinical_diagnosis": "Хронический синусит",
            "mis_diagnos": "J32.9",
            "patient_age_years": 38,
            "visit_date": "2026-07-15",
        },
        "record": {"visit_id": "eval_j32", "specialty": "ЛОР", "date": "2026-07-15"},
        "expect_any": ["синусит"],
        "reject_any": ["стволов", "гемопоэтическ", "экстренн"],
    },
    {
        "id": "af_i48",
        "clinical": {
            "clinical_diagnosis": "Фибрилляция предсердий",
            "mis_diagnos": "I48.0",
            "patient_age_years": 70,
            "visit_date": "2026-07-15",
        },
        "record": {"visit_id": "eval_i48", "specialty": "Кардиолог", "date": "2026-07-15"},
        "expect_any": ["фибрилляц", "предсерд"],
        "reject_any": ["стволов", "гемопоэтическ", "экстренн"],
    },
]


def _reset_caches() -> None:
    from clinical_knowledge.ilex_protocol_passports import clear_ilex_passport_cache
    from clinical_knowledge.loader import clear_clinical_knowledge_cache
    from clinical_knowledge.protocol_candidate_index import clear_candidate_index

    clear_ilex_passport_cache()
    clear_clinical_knowledge_cache()
    clear_candidate_index()


def _blob(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in ("title", "condition_label", "source_path", "matched_condition")
    ).lower()


def _run_suggest(case: dict[str, Any]) -> dict[str, Any]:
    from clinical_knowledge.case_protocol_suggest import suggest_protocols_for_case

    os.environ["CASE_PROTOCOL_SUGGEST"] = "1"
    result = suggest_protocols_for_case(
        clinical=case["clinical"],
        record=case["record"],
        limit=3,
    )
    items = list(result.get("items") or [])
    top = items[0] if items else {}
    blob = _blob(top)
    expect = list(case.get("expect_any") or [])
    reject = list(case.get("reject_any") or [])
    hit = (not expect) or any(needle in blob for needle in expect)
    bad = any(needle in blob for needle in reject)
    return {
        "id": case["id"],
        "available": bool(result.get("available")),
        "hit": bool(hit and not bad and items),
        "wrong": bool(bad),
        "title": str(top.get("title") or "")[:160],
        "path": str(top.get("source_path") or "")[-120:],
        "score": top.get("score") or top.get("match_score"),
    }


def _golden_ok() -> dict[str, Any]:
    from clinical_knowledge.mo_kp_suggest_golden_eval import (
        evaluate_mo_kp_suggest_row,
        load_mo_kp_suggest_golden,
    )

    rows = load_mo_kp_suggest_golden()
    failed: list[str] = []
    for row in rows:
        out = evaluate_mo_kp_suggest_row(row)
        if not out.get("ok"):
            failed.append(str(row.get("id") or "?"))
    return {"n": len(rows), "failed": failed, "ok": not failed}


def _passport_hits() -> list[dict[str, Any]]:
    from clinical_knowledge.ilex_protocol_passports import match_passports_for_diagnosis

    out: list[dict[str, Any]] = []
    for case in NOSOLOGY_CASES:
        dx = str(case["clinical"].get("clinical_diagnosis") or "")
        icd = [str(case["clinical"].get("mis_diagnos") or "")]
        rows = match_passports_for_diagnosis(dx, icd_codes=icd, audience="adult", limit=3)
        top = rows[0] if rows else {}
        blob = str(top.get("protocol_title") or "").lower()
        expect = list(case.get("expect_any") or [])
        out.append(
            {
                "id": case["id"],
                "hit": bool(rows) and any(n in blob for n in expect),
                "title": str(top.get("protocol_title") or "")[:160],
                "icd": list(top.get("icd10_primary") or [])[:8],
                "score": top.get("match_score"),
            }
        )
    return out


def main() -> int:
    os.environ.setdefault("CASE_PROTOCOL_SUGGEST", "1")
    os.environ["ILEX_PASSPORTS"] = "0"
    _reset_caches()
    off_cases = [_run_suggest(case) for case in NOSOLOGY_CASES]
    off_gold = _golden_ok()

    os.environ["ILEX_PASSPORTS"] = "1"
    _reset_caches()
    on_cases = [_run_suggest(case) for case in NOSOLOGY_CASES]
    on_gold = _golden_ok()
    passports = _passport_hits()

    report = {
        "passport_direct": passports,
        "suggest_overlay_off": off_cases,
        "suggest_overlay_on": on_cases,
        "golden_off": {"n": off_gold["n"], "failed": off_gold["failed"]},
        "golden_on": {"n": on_gold["n"], "failed": on_gold["failed"]},
        "nosology_hit_off": sum(1 for row in off_cases if row["hit"]),
        "nosology_hit_on": sum(1 for row in on_cases if row["hit"]),
        "wrong_off": sum(1 for row in off_cases if row["wrong"]),
        "wrong_on": sum(1 for row in on_cases if row["wrong"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["nosology_hit_on"] < report["nosology_hit_off"]:
        return 2
    if report["wrong_on"] > report["wrong_off"]:
        return 3
    if on_gold["failed"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
