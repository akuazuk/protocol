#!/usr/bin/env python3
"""Пересчитать deep/findings в kz_l1_*_cases.jsonl новым кодом (без МИС).

Клинический текст берётся из дневного CSV ``mo_YYYY-MM-DD.csv``
(в cases.jsonl его обычно нет - только scores/meta).

Пример на Render:
  .venv/bin/python scripts/rescore_mo_deep_days.py \\
    --data-root /var/data/medical_exams \\
    --first-date 2026-08-01 --last-date 2026-08-05
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.kz_deep_eval import evaluate_kz_deep, load_drug_ctx  # noqa: E402

_CLINICAL_KEYS = (
    "complaints",
    "anamnesis_doctor",
    "anamnesis_auto",
    "objective_status",
    "clinical_diagnosis",
    "exam_data",
    "exam_recommendations",
    "treatment_recommendations",
    "manipulations",
    "mis_diagnos",
    "mkb_code_main",
    "diagnosis_structured_raw",
    "return_date",
    "doctor_specialization",
    "specialty_id",
    "visit_date_text",
    "filial",
)


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def _load_csv_by_visit(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            vid = str(row.get("visit_id") or "").strip()
            if vid:
                out[vid] = row
    return out


def _case_for_deep(row: dict[str, Any], csv_row: dict[str, str] | None) -> dict[str, Any]:
    """Собрать кейс для deep: meta из jsonl + клинические поля из CSV."""
    out = dict(row)
    clinical = row.get("clinical") if isinstance(row.get("clinical"), dict) else {}
    for key, value in clinical.items():
        if value and not out.get(key):
            out[key] = value
    if csv_row:
        for key in _CLINICAL_KEYS:
            value = csv_row.get(key)
            if value not in (None, ""):
                out[key] = value
        # служебные id из среза, если в cases пусто
        for key in ("mis_id", "patient_id", "doctor_fio", "pay_type"):
            value = csv_row.get(key)
            if value not in (None, "") and not out.get(key):
                out[key] = value
    return out


def _compact_deep(deep: dict[str, Any]) -> dict[str, Any]:
    """Как в run_mis_protocol_l1_batch: не раздувать jsonl полным dump."""
    return {
        "axes": deep.get("axes"),
        "overall_pct": deep.get("overall_pct"),
        "status": deep.get("overall_status") or deep.get("status"),
        "overall_status": deep.get("overall_status"),
        "n_findings": deep.get("n_findings"),
        "n_by_severity": deep.get("n_by_severity"),
        "has_potential_harm": deep.get("has_potential_harm"),
        "protocol_used": deep.get("protocol_used"),
        "findings": (deep.get("findings") or [])[:20],
        "shadow_findings": (deep.get("shadow_findings") or [])[:20],
        "reg55": deep.get("reg55"),
    }


def rescore_day(
    day: date,
    *,
    data_root: Path,
    update_primary_score: bool = False,
) -> dict[str, Any]:
    secure = data_root / "secure_cases" / f"{day:%Y}" / f"{day:%m}"
    path = secure / f"kz_l1_{day.isoformat()}_cases.jsonl"
    if not path.is_file():
        return {"date": day.isoformat(), "status": "missing_cases"}
    csv_by_visit = _load_csv_by_visit(secure / f"mo_{day.isoformat()}.csv")
    if not csv_by_visit:
        return {
            "date": day.isoformat(),
            "status": "missing_csv",
            "cases_path": str(path),
        }
    drug_ctx = load_drug_ctx()
    rows: list[dict[str, Any]] = []
    changed = 0
    joined = 0
    primary_updated = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = str(row.get("visit_id") or row.get("case_id") or "").strip()
        csv_row = csv_by_visit.get(vid)
        if csv_row:
            joined += 1
        before = json.dumps(row.get("deep") or {}, ensure_ascii=False, sort_keys=True)
        deep = evaluate_kz_deep(
            _case_for_deep(row, csv_row),
            protocol_ctx=None,
            drug_ctx=drug_ctx,
        )
        # Не трогаем overall_pct / evaluation_v4 по умолчанию - primary score живёт отдельно.
        row["deep"] = _compact_deep(deep)
        if update_primary_score:
            before_primary = (row.get("overall_pct"), row.get("status"))
            if deep.get("overall_pct") is not None:
                row["overall_pct"] = deep.get("overall_pct")
            status = deep.get("overall_status") or deep.get("status")
            if status:
                row["status"] = status
            if before_primary != (row.get("overall_pct"), row.get("status")):
                primary_updated += 1
        after = json.dumps(row.get("deep") or {}, ensure_ascii=False, sort_keys=True)
        if before != after:
            changed += 1
        rows.append(row)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    out = {
        "date": day.isoformat(),
        "status": "success",
        "cases": len(rows),
        "joined_csv": joined,
        "changed": changed,
    }
    if update_primary_score:
        out["primary_updated"] = primary_updated
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--first-date", type=date.fromisoformat, required=True)
    ap.add_argument("--last-date", type=date.fromisoformat, required=True)
    ap.add_argument(
        "--update-primary-score",
        action="store_true",
        help="Также обновить overall_pct/status кейса из deep (по умолчанию только deep-блок).",
    )
    args = ap.parse_args(argv)
    results = [
        rescore_day(
            day,
            data_root=args.data_root.expanduser(),
            update_primary_score=bool(args.update_primary_score),
        )
        for day in _days(args.first_date, args.last_date)
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] in {"success", "missing_cases"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
