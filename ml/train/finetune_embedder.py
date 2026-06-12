#!/usr/bin/env python3
"""Fine-tune bi-encoder для RAG (заглушка CLI).

Реализация: LoRA на intfloat/multilingual-e5-small или BAAI/bge-m3.
Требует: pip install sentence-transformers peft (отдельный venv).

  python3 ml/train/finetune_embedder.py --dataset ../datasets/retrieval_pairs.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Protocol embedder (stub)")
    parser.add_argument("--dataset", type=Path, default=Path("ml/datasets/retrieval_pairs.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("ml/configs/default.json"))
    parser.add_argument("--output", type=Path, default=Path("ml/registry/checkpoints/embedder-v1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}. Run: python3 scripts/export_training_feedback.py")

    n = sum(1 for _ in args.dataset.open(encoding="utf-8") if _.strip())
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    base = cfg["models"]["embedder"]["base"]
    print(f"Would fine-tune {base} on {n} pairs from {args.dataset}")
    print(f"Output: {args.output}")
    if args.dry_run:
        print("Dry run OK. Install sentence-transformers + peft for actual training.")
        return
    raise SystemExit(
        "Training not wired in repo yet. Use --dry-run to validate dataset. "
        "See ml/README.md for MLOps roadmap."
    )


if __name__ == "__main__":
    main()
