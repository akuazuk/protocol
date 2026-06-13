#!/usr/bin/env python3
"""Извлечь правила из всех гастро-PDF в chunks.jsonl → data/gastro_mvp/rules/auto_*.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.rules_from_corpus import (
    extract_rules_all_gastro_pdfs,
    merge_rules_into_gastro_mvp,
)

REGISTRY = ROOT / "data" / "gastro_mvp" / "protocol_registry.jsonl"
CHUNKS = ROOT / "output" / "chunks" / "chunks.jsonl"


def main() -> None:
    if not CHUNKS.is_file():
        print(f"WARN: нет {CHUNKS} - сначала run_pipeline")
        sys.exit(1)
    if not REGISTRY.is_file():
        print(f"WARN: нет {REGISTRY} - сначала build_protocol_cards.py")
        sys.exit(1)

    extracted, meta = extract_rules_all_gastro_pdfs(CHUNKS, REGISTRY)
    counts = merge_rules_into_gastro_mvp(extracted, ROOT / "data" / "gastro_mvp" / "rules")

    pdfs_with_rules = sum(
        1 for v in meta.get("pdfs", {}).values() if isinstance(v, dict) and v.get("rules", 0) > 0
    )
    summary = {
        "pdfs_total": meta.get("pdfs_total"),
        "pdfs_with_rules": pdfs_with_rules,
        "rules_by_condition": counts,
        "total_rules": sum(counts.values()),
    }
    out = ROOT / "data" / "gastro_mvp" / "rules_extraction_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
