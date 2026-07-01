#!/usr/bin/env python3
"""Патч indexable/noise_flags в существующих corpus_chunks_parts (Фаза 3).

Применяет apply_indexable_flags без полного rebuild PDF.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.chunk_quality import apply_indexable_flags  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="corpus_chunks_parts/chunks.part.*.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = patched = dropped = 0
    for path in sorted(ROOT.glob(args.glob)):
        lines_out: list[str] = []
        file_patched = 0
        file_dropped = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ch = json.loads(line)
                except json.JSONDecodeError:
                    lines_out.append(line)
                    continue
                total += 1
                before = ch.get("indexable")
                apply_indexable_flags(ch)
                after = ch.get("indexable")
                if before is not False and after is False:
                    file_dropped += 1
                if ch.get("noise_flags") or before != after:
                    file_patched += 1
                lines_out.append(json.dumps(ch, ensure_ascii=False) + "\n")
        patched += file_patched
        dropped += file_dropped
        if not args.dry_run:
            path.write_text("".join(lines_out), encoding="utf-8")
        print(f"{path.name}: patched={file_patched} newly_nonindexable={file_dropped}")
    print(f"TOTAL chunks={total} patched={patched} newly_nonindexable={dropped} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
