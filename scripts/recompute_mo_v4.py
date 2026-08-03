#!/usr/bin/env python3
"""Recompute MO case artifacts with scorer v4 without overwriting v3 input."""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.kz_deep_eval import load_drug_ctx, resolve_protocol_ctx
from clinical_knowledge.kz_evaluation_v4 import evaluate_kz_v4


def _score(row: dict[str, Any]) -> float | None:
    value = (row.get("deep") or {}).get("overall_pct")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def recompute(input_path: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    by_day: dict[str, list[float]] = defaultdict(list)
    by_specialty: dict[str, list[float]] = defaultdict(list)
    largest: list[dict[str, Any]] = []
    total = scored = fallback = 0
    drug_ctx = load_drug_ctx()
    with input_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            previous = _score(row)
            try:
                protocol = resolve_protocol_ctx(row)
                result = evaluate_kz_v4(
                    row,
                    protocol_ctx=protocol,
                    drug_ctx=drug_ctx,
                    legacy={
                        "deep_overall_pct": previous,
                        "deep_status": (row.get("deep") or {}).get("status"),
                        "l1_overall_pct": row.get("overall_pct"),
                        "v3_score_pct": (row.get("evaluation_v3") or {}).get("score_pct"),
                    },
                )
                row["evaluation_v4"] = result.to_public_dict()
                row["overall_pct_v3"] = previous
                row["overall_pct"] = result.score_pct
                row["status"] = result.status
                row["scorer_version"] = result.scorer_version
                scored += result.score_pct is not None
                if previous is not None and result.score_pct is not None:
                    delta = round(result.score_pct - previous, 2)
                    day = str(row.get("date") or row.get("visit_date") or "")[:10]
                    specialty = str(
                        row.get("doctor_specialization") or row.get("specialty") or "Не указано"
                    )
                    by_day[day].append(delta)
                    by_specialty[specialty].append(delta)
                    largest.append(
                        {
                            "case_id": str(row.get("mis_id") or row.get("visit_id") or ""),
                            "date": day,
                            "specialty": specialty,
                            "v3": previous,
                            "v4": result.score_pct,
                            "delta": delta,
                            "reasons": result.attention_reasons + result.risk.reasons,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                row["evaluation_v4_error"] = str(exc)[:300]
                row["scorer_version"] = "deep-v2-fallback"
                fallback += 1
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output_path)

    def aggregates(groups: dict[str, list[float]]) -> list[dict[str, Any]]:
        return [
            {
                "key": key,
                "n": len(values),
                "mean_delta": round(statistics.fmean(values), 2),
                "median_delta": round(statistics.median(values), 2),
            }
            for key, values in sorted(groups.items())
            if values
        ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "total": total,
        "scored_v4": scored,
        "fallback": fallback,
        "by_day": aggregates(by_day),
        "by_specialty": aggregates(by_specialty),
        "largest_changes": sorted(largest, key=lambda item: abs(item["delta"]), reverse=True)[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warehouse", type=Path)
    args = parser.parse_args()
    if args.warehouse and args.warehouse.is_file():
        backup = args.warehouse.with_name(
            f"{args.warehouse.stem}.before-v4-{datetime.now():%Y%m%d-%H%M%S}.sqlite"
        )
        shutil.copy2(args.warehouse, backup)
        print(f"warehouse_backup={backup}")
    report = recompute(args.input, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("total", "scored_v4", "fallback")}))
    return 0 if report["fallback"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
