#!/usr/bin/env python3
"""Построить data/catalog/protocol_icd_profiles.jsonl из rich_chunks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from clinical_knowledge.protocol_icd_profile_index import build_protocol_icd_profile_index

    summary = build_protocol_icd_profile_index()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
