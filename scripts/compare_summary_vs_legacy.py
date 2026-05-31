#!/usr/bin/env python3
"""Compare legacy vs summary vs hybrid on consultation fixtures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.protocol_summary.summary_compare import (  # noqa: E402
    append_batch_csv,
    compare_modes_on_text,
    write_comparison_report,
)

FIXTURES = ROOT / "tests" / "fixtures" / "consultations"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default=None)
    ap.add_argument("--output", type=str, default="data/reports/method_comparison")
    args = ap.parse_args()
    out_dir = Path(args.output)
    rows = []
    files = [Path(args.file)] if args.file else sorted(FIXTURES.glob("*.txt"))
    for path in files:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        cmp = compare_modes_on_text(text, consultation_id=path.stem)
        write_comparison_report(cmp, out_dir / f"{path.stem}.md", consultation_id=path.stem)
        rows.append({
            "consultation_id": path.stem,
            "legacy_score": cmp.get("legacy_score"),
            "summary_score": cmp.get("summary_score"),
            "hybrid_score": cmp.get("hybrid_score"),
            "score_delta": cmp.get("score_delta_summary"),
            "same_decision": cmp.get("same_decision_legacy_summary"),
        })
        print(f"{path.stem}: legacy={cmp.get('legacy_score')} summary={cmp.get('summary_score')} hybrid={cmp.get('hybrid_score')}")
    append_batch_csv(rows, out_dir / "batch_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
