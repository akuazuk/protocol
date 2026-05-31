#!/usr/bin/env python3
"""Отчёт покрытия автоизвлечения правил по гастро-PDF."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.rules_from_corpus import extract_rules_all_gastro_pdfs

REGISTRY = ROOT / "data" / "gastro_mvp" / "protocol_registry.jsonl"
CHUNKS = ROOT / "output" / "chunks" / "chunks.jsonl"
OUT = ROOT / "data" / "gastro_mvp" / "rules_coverage_report.json"


def main() -> int:
    if not CHUNKS.is_file() or not REGISTRY.is_file():
        print("WARN: нужны chunks.jsonl и protocol_registry.jsonl")
        return 1

    extracted, meta = extract_rules_all_gastro_pdfs(CHUNKS, REGISTRY)
    pdfs = meta.get("pdfs") or {}
    with_rules = [sp for sp, info in pdfs.items() if isinstance(info, dict) and info.get("rules", 0) > 0]
    without_rules = [sp for sp, info in pdfs.items() if isinstance(info, dict) and info.get("rules", 0) == 0]

    report = {
        "pdfs_total": meta.get("pdfs_total"),
        "pdfs_with_rules": len(with_rules),
        "pdfs_without_rules": len(without_rules),
        "rules_by_condition": {cid: len(rules) for cid, rules in extracted.items()},
        "total_rules": sum(len(v) for v in extracted.values()),
        "with_rules": [sp.replace("\\", "/") for sp in with_rules],
        "without_rules": [sp.replace("\\", "/") for sp in without_rules],
        "per_pdf": pdfs,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "pdfs_with_rules": report["pdfs_with_rules"],
            "pdfs_without_rules": report["pdfs_without_rules"],
            "total_rules": report["total_rules"],
            "rules_by_condition": report["rules_by_condition"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
