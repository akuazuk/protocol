#!/usr/bin/env python3
"""Validate Protocol Summary YAML/JSON on disk."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.protocol_summary.loader import load_protocol_summaries  # noqa: E402
from clinical_knowledge.protocol_summary.validator import (  # noqa: E402
    validate_protocol_summary,
    write_validation_report,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    invalid = 0
    for s in load_protocol_summaries():
        r = validate_protocol_summary(s, strict=args.strict)
        write_validation_report(s, r)
        if r.status == "invalid":
            invalid += 1
        print(f"{s.protocol_id}: {r.status}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
