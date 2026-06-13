#!/usr/bin/env python3
"""Копирует последние N снимков из manifest Render → fixtures для git.

Пример (Render Shell или локально):
  python3 scripts/sync_consult_fixtures.py
  python3 scripts/sync_consult_fixtures.py --last 20 --out tests/fixtures/consult_replay.jsonl

Дальше: git add tests/fixtures/consult_replay.jsonl && git push
        В Cursor: git pull && python3 scripts/replay_consult_archive.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.analysis_archive import archive_dir, load_snapshots  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync consult manifest → git fixtures")
    ap.add_argument("--last", type=int, default=50, help="How many recent snapshots")
    ap.add_argument(
        "--out",
        type=str,
        default="tests/fixtures/consult_replay.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Override manifest path (default: output/consult_archive/manifest.jsonl)",
    )
    args = ap.parse_args()

    if args.manifest:
        # временно подменяем через env не нужно - читаем файл напрямую
        manifest = Path(args.manifest)
        if not manifest.is_file():
            print(f"Manifest not found: {manifest}", file=sys.stderr)
            return 1
        snaps = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
        snaps = snaps[-args.last :]
    else:
        snaps = load_snapshots(limit=args.last)
        if not snaps:
            print(
                f"No snapshots in {archive_dir() / 'manifest.jsonl'}\n"
                "Enable CONSULT_ARCHIVE_ANALYSES=1 on Render or pass --manifest",
                file=sys.stderr,
            )
            return 1

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in snaps:
            f.write(json.dumps(s, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {len(snaps)} snapshot(s) → {out.relative_to(ROOT)}")
    print("Next: git add/commit/push → git pull in Cursor → replay_consult_archive.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
