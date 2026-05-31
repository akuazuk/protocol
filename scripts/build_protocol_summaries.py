#!/usr/bin/env python3
"""Build draft Protocol Summary YAML from protocol_cards."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.protocol_summary.builder import build_protocol_summaries  # noqa: E402
from clinical_knowledge.protocol_summary.loader import clear_protocol_summary_cache  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max PDF count")
    ap.add_argument("--rubric", type=str, default=None, help="specialty_slug filter")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--no-rag", action="store_true")
    args = ap.parse_args()
    summaries = build_protocol_summaries(
        limit=args.limit,
        rubric=args.rubric,
        publish=not args.no_publish,
        export_rag=not args.no_rag,
    )
    clear_protocol_summary_cache()
    print(f"Built {len(summaries)} protocol summaries (1 per PDF)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
