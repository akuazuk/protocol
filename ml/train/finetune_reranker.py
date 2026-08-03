#!/usr/bin/env python3
"""Fine-tune cross-encoder reranker для подбора протоколов (S5).

Учит cross-encoder (query, passage) -> relevance на парах из retrieval-фидбэка.
Позитивы: `positive_text` из retrieval_pairs_resolved.jsonl.
Негативы: пассажи других протоколов (in-corpus mining).

Gate/eval: reranking на holdout (accuracy@1, MRR@10) vs базовая модель.

Пример:
  python3 ml/train/finetune_reranker.py --dry-run
  python3 ml/train/finetune_reranker.py --epochs 2 --num-negatives 4
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.eval.eval_embedder_mrr import split_by_path_fold  # noqa: E402

DEFAULT_DATASET = ROOT / "ml/datasets/retrieval_pairs_resolved.jsonl"
DEFAULT_OUT = ROOT / "ml/registry/checkpoints/reranker-v1"
REPORT_DIR = ROOT / "ml/experiments/reranker_v1"
MAX_PASSAGE_CHARS = 1200


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("query") and row.get("positive_text"):
            row["positive_text"] = str(row["positive_text"])[:MAX_PASSAGE_CHARS]
            rows.append(row)
    return rows


def _mine_negatives(
    rows: list[dict], idx: int, pool_texts: list[str], pool_paths: list[str], k: int, rng: random.Random
) -> list[str]:
    """k пассажей других протоколов (разный positive_path) как негативы."""
    own_path = rows[idx].get("positive_path")
    candidates = [i for i in range(len(pool_texts)) if pool_paths[i] != own_path]
    rng.shuffle(candidates)
    return [pool_texts[i] for i in candidates[:k]]


def build_examples(rows: list[dict], num_negatives: int, seed: int):
    from sentence_transformers import InputExample

    rng = random.Random(seed)
    pool_texts = [r["positive_text"] for r in rows]
    pool_paths = [r.get("positive_path") for r in rows]
    examples = []
    for i, r in enumerate(rows):
        q = r["query"]
        examples.append(InputExample(texts=[q, r["positive_text"]], label=1.0))
        for neg in _mine_negatives(rows, i, pool_texts, pool_paths, num_negatives, rng):
            examples.append(InputExample(texts=[q, neg], label=0.0))
    rng.shuffle(examples)
    return examples


def eval_reranking(model, holdout: list[dict], all_rows: list[dict], num_cand: int, seed: int) -> dict:
    """Для каждого holdout-запроса ранжируем [позитив + негативы]; @1 и MRR."""
    rng = random.Random(seed + 7)
    pool_texts = [r["positive_text"] for r in all_rows]
    pool_paths = [r.get("positive_path") for r in all_rows]
    hits1 = 0
    rr_sum = 0.0
    n = 0
    for r in holdout:
        q = r["query"]
        pos = r["positive_text"]
        negs = _mine_negatives(all_rows, all_rows.index(r) if r in all_rows else 0,
                               pool_texts, pool_paths, num_cand - 1, rng)
        if not negs:
            continue
        cands = [pos] + negs
        pairs = [[q, c] for c in cands]
        scores = model.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(cands)), key=lambda i: float(scores[i]), reverse=True)
        rank = order.index(0) + 1  # позитив имеет индекс 0
        hits1 += int(rank == 1)
        rr_sum += 1.0 / rank
        n += 1
    return {
        "accuracy_at_1": round(hits1 / n, 4) if n else 0.0,
        "mrr": round(rr_sum / n, 4) if n else 0.0,
        "n": n,
        "candidates_per_query": num_cand,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Protocol reranker (S5)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=ROOT / "ml/configs/default.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--eval-candidates", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fold", type=int, default=0, help="0..n-1 hold-out; -1 = train on all")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}. Run export_training_feedback.py")

    all_rows = load_rows(args.dataset)
    print(f"Dataset rows with text: {len(all_rows)}")
    if not all_rows:
        raise SystemExit("No trainable rows")
    if args.dry_run:
        print("Dry run OK.")
        return

    cfg = json.loads(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
    model_name = args.base_model or (cfg.get("models") or {}).get("reranker", {}).get(
        "base", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    train_rows, holdout_rows = split_by_path_fold(all_rows, args.test_fold, args.n_folds)
    examples = build_examples(train_rows, args.num_negatives, args.seed)

    import time

    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader

    # baseline (до обучения) на holdout
    base_model = CrossEncoder(model_name, num_labels=1)
    base_eval = eval_reranking(base_model, holdout_rows, all_rows, args.eval_candidates, args.seed)

    model = CrossEncoder(model_name, num_labels=1)
    train_dl = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    warmup = max(10, int(len(examples) / max(args.batch_size, 1)))

    t0 = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_dl,
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
    )
    train_sec = time.perf_counter() - t0
    model.save_pretrained(str(args.output))

    # eval на обученной модели в памяти (перезагрузка чекпоинта в ST 5.x нестабильна)
    tuned_eval = eval_reranking(model, holdout_rows, all_rows, args.eval_candidates, args.seed)

    gate_pass = tuned_eval["accuracy_at_1"] >= base_eval["accuracy_at_1"] and tuned_eval["mrr"] >= base_eval["mrr"]
    meta = {
        "base_model": model_name,
        "train_examples": len(examples),
        "train_pairs_pos": len(train_rows),
        "num_negatives": args.num_negatives,
        "holdout_pairs": len(holdout_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_seconds": round(train_sec, 2),
        "baseline_eval": base_eval,
        "tuned_eval": tuned_eval,
        "gate_pass": gate_pass,
        "output": str(args.output),
    }
    (args.output / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = [
        "# Reranker v1 (S5)",
        "",
        f"- Base: `{model_name}`",
        f"- Train examples: {len(examples)} ({len(train_rows)} pos x1 + {args.num_negatives} neg)",
        f"- Holdout pairs: {len(holdout_rows)} (fold {args.test_fold}/{args.n_folds})",
        f"- Train time: **{train_sec:.1f}s** (CPU)",
        "",
        "## Reranking eval (holdout)",
        "",
        "| | accuracy@1 | MRR | n |",
        "|---|---|---|---|",
        f"| base | {base_eval['accuracy_at_1']} | {base_eval['mrr']} | {base_eval['n']} |",
        f"| tuned | {tuned_eval['accuracy_at_1']} | {tuned_eval['mrr']} | {tuned_eval['n']} |",
        "",
        f"**Gate (tuned >= base):** {'PASS' if gate_pass else 'FAIL'}",
        "",
    ]
    (args.report_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
