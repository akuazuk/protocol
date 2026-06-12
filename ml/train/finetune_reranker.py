#!/usr/bin/env python3
"""Fine-tune cross-encoder reranker (заглушка CLI)."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("ml/datasets/retrieval_pairs.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dataset.is_file():
        raise SystemExit("Run: python3 scripts/export_training_feedback.py")
    n = sum(1 for _ in args.dataset.open(encoding="utf-8") if _.strip())
    print(f"Reranker stub: {n} pairs. See ml/README.md")
    if not args.dry_run:
        raise SystemExit("Use --dry-run")


if __name__ == "__main__":
    main()
