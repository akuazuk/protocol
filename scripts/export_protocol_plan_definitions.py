#!/usr/bin/env python3
"""Экспорт Protocol Summary Cards → FHIR PlanDefinition JSON (bundle)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "plan_definitions_bundle.json",
        help="Путь к JSON Bundle",
    )
    parser.add_argument("--status", default="draft", choices=("draft", "active", "retired"))
    parser.add_argument("--limit", type=int, default=0, help="Макс. PlanDefinition (0 = все usable)")
    args = parser.parse_args()

    from clinical_knowledge.protocol_summary.loader import load_protocol_summaries
    from clinical_knowledge.protocol_summary.plan_definition_export import (
        export_summaries_to_plan_definitions,
    )

    summaries = list(load_protocol_summaries(usable_only=True))
    pds = export_summaries_to_plan_definitions(summaries, usable_only=True, status=args.status)
    if args.limit > 0:
        pds = pds[: args.limit]

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": pd} for pd in pds],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(pds)} PlanDefinition resources to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
