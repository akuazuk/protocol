#!/usr/bin/env python3
"""Аудит меток rich-чанков: preamble, signal, obligation, passport."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl"
META_DIR = ROOT / "output" / "rich_meta"


def audit_chunks(path: Path, *, limit: int = 0) -> dict:
    stats: Counter[str] = Counter()
    samples: dict[str, list[str]] = {"low_signal": [], "preamble": [], "inpatient_only": []}
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if limit and n >= limit:
                break
            row = json.loads(line)
            n += 1
            tags = row.get("tags") or {}
            stats[f"signal:{tags.get('signal', '?')}"] += 1
            stats[f"obligation:{tags.get('obligation', '?')}"] += 1
            if tags.get("is_preamble"):
                stats["preamble"] += 1
                if len(samples["preamble"]) < 5:
                    samples["preamble"].append(str(row.get("chunk_id")))
            if tags.get("signal") == "low":
                stats["low_signal"] += 1
                if len(samples["low_signal"]) < 5:
                    samples["low_signal"].append(str(row.get("chunk_id")))
            care = tags.get("care_setting") or row.get("care_setting") or []
            if "inpatient" in care and "ambulatory" not in care:
                stats["inpatient_only"] += 1
                if len(samples["inpatient_only"]) < 5:
                    samples["inpatient_only"].append(str(row.get("chunk_id")))
    return {"chunks_read": n, "stats": dict(stats), "samples": samples}


def audit_meta(dir_path: Path) -> dict:
    stats: Counter[str] = Counter()
    n = 0
    for fp in sorted(dir_path.glob("*.json")):
        try:
            meta = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        n += 1
        pt = meta.get("protocol_tags") or meta.get("tags") or {}
        if pt.get("admin_order"):
            stats["admin_order"] += 1
        if pt.get("usable_for_kz_review"):
            stats["usable_for_kz_review"] += 1
        stats[f"cluster:{pt.get('condition_cluster', '?')}"] += 1
    return {"meta_files": n, "stats": dict(stats)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit chunk tags in rich corpus")
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out: dict = {}
    if args.chunks.exists():
        out["chunks"] = audit_chunks(args.chunks, limit=args.limit)
    else:
        out["chunks"] = {"error": f"not found: {args.chunks}"}

    if META_DIR.exists():
        out["meta"] = audit_meta(META_DIR)
    else:
        out["meta"] = {"error": f"not found: {META_DIR}"}

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
