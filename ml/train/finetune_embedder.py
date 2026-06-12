#!/usr/bin/env python3
"""Fine-tune bi-encoder для RAG (e5 / sentence-transformers)."""
from __future__ import annotations

import os

# На macOS Trainer может уйти в MPS и упасть по памяти — принудительно CPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.eval.eval_embedder_mrr import split_by_path_fold  # noqa: E402


def _e5_query(q: str) -> str:
    return f"query: {q.strip()}"


def _e5_passage(p: str) -> str:
    return f"passage: {p.strip()}"


def load_train_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("query") and row.get("positive_text"):
            rows.append(row)
    return rows


def train(
    *,
    dataset: Path,
    config: Path,
    output: Path,
    base_model: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    test_fold: int,
    n_folds: int,
    max_train: int,
    device: str,
) -> dict:
    cfg = json.loads(config.read_text(encoding="utf-8")) if config.is_file() else {}
    model_name = base_model or (cfg.get("models") or {}).get("embedder", {}).get(
        "base", "intfloat/multilingual-e5-small"
    )

    all_rows = load_train_rows(dataset)
    if not all_rows:
        raise SystemExit(f"No trainable rows in {dataset}")

    train_rows, holdout_rows = split_by_path_fold(all_rows, test_fold, n_folds)
    if max_train > 0:
        random.seed(seed)
        train_rows = train_rows[:max_train]

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    examples = [
        InputExample(texts=[_e5_query(r["query"]), _e5_passage(r["positive_text"])])
        for r in train_rows
    ]

    model = SentenceTransformer(model_name, device=device)
    train_dl = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    output.mkdir(parents=True, exist_ok=True)
    warmup = max(10, int(len(examples) / max(batch_size, 1)))
    model.fit(
        train_objectives=[(train_dl, train_loss)],
        epochs=epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": lr},
        output_path=str(output),
        show_progress_bar=True,
    )

    meta = {
        "base_model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "train_pairs": len(train_rows),
        "holdout_pairs": len(holdout_rows),
        "test_fold": test_fold,
        "n_folds": n_folds,
        "output": str(output),
        "device": device,
    }
    (output / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Protocol embedder")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("ml/datasets/retrieval_pairs_resolved.jsonl"),
    )
    parser.add_argument("--config", type=Path, default=Path("ml/configs/default.json"))
    parser.add_argument("--output", type=Path, default=Path("ml/registry/checkpoints/embedder-v1"))
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fold", type=int, default=0, help="0..n-1 hold-out; -1 = train on all")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=0, help="0 = all train fold rows")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}. Run export_training_feedback.py")

    n = len(load_train_rows(args.dataset))
    print(f"Dataset rows with text: {n}")
    if args.dry_run:
        print("Dry run OK.")
        return

    meta = train(
        dataset=args.dataset,
        config=args.config,
        output=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        test_fold=args.test_fold,
        n_folds=args.n_folds,
        max_train=args.max_train,
        device=args.device,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
