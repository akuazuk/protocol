#!/usr/bin/env python3
"""MRR@k для bi-encoder: closed-set ranking по корпусу passage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.chunk_resolver import build_path_index, resolve_path_text  # noqa: E402


def _e5_query(q: str) -> str:
    return f"query: {q.strip()}"


def _e5_passage(p: str) -> str:
    return f"passage: {p.strip()}"


def load_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_corpus_from_pairs(
    pairs: list[dict],
    path_index: dict[str, str],
) -> tuple[list[str], dict[str, int]]:
    """Уникальные passage по path; возвращает тексты и path→idx."""
    path_to_idx: dict[str, int] = {}
    passages: list[str] = []
    for row in pairs:
        path = (row.get("positive_path") or "").strip()
        text = (row.get("positive_text") or "").strip()
        if not path:
            continue
        if path in path_to_idx:
            continue
        if not text:
            text = resolve_path_text(path, path_index) or ""
        if not text:
            continue
        path_to_idx[path] = len(passages)
        passages.append(text)
    return passages, path_to_idx


def mrr_at_k(
    model,
    queries: list[dict],
    passages: list[str],
    path_to_idx: dict[str, int],
    *,
    k: int = 10,
    batch_size: int = 64,
) -> dict:
    if not queries or not passages:
        return {"mrr_at_k": 0.0, "recall_at_1": 0.0, "n": 0, "k": k}

    import numpy as np

    p_emb = model.encode(
        [_e5_passage(p) for p in passages],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    ranks: list[int] = []
    hits1 = 0
    skipped = 0
    for row in queries:
        path = (row.get("positive_path") or "").strip()
        qtext = (row.get("query") or "").strip()
        if not path or not qtext or path not in path_to_idx:
            skipped += 1
            continue
        target = path_to_idx[path]
        q_emb = model.encode(
            [_e5_query(qtext)],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        scores = p_emb @ q_emb
        order = np.argsort(-scores)
        rank = int(np.where(order == target)[0][0]) + 1
        ranks.append(rank)
        if rank == 1:
            hits1 += 1

    n = len(ranks)
    mrr = sum(1.0 / r for r in ranks) / n if n else 0.0
    return {
        "mrr_at_k": round(mrr, 4),
        "recall_at_1": round(hits1 / n, 4) if n else 0.0,
        "n": n,
        "skipped": skipped,
        "k": k,
        "corpus_size": len(passages),
    }


def path_fold(path: str, n_folds: int = 5) -> int:
    return abs(hash(path)) % n_folds


def split_by_path_fold(
    pairs: list[dict],
    fold: int,
    n_folds: int = 5,
) -> tuple[list[dict], list[dict]]:
    if fold < 0:
        return list(pairs), []
    train, test = [], []
    for row in pairs:
        p = (row.get("positive_path") or "").strip()
        if not p:
            continue
        if path_fold(p, n_folds) == fold:
            test.append(row)
        else:
            train.append(row)
    return train, test


def main() -> int:
    parser = argparse.ArgumentParser(description="MRR eval for Protocol embedder")
    parser.add_argument("--dataset", type=Path, default=ROOT / "ml/datasets/retrieval_pairs_resolved.jsonl")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-small")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=-1, help="0..4 for CV fold test; -1 = all rows")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    pairs = [r for r in load_pairs(args.dataset) if r.get("positive_text")]
    if not pairs:
        raise SystemExit("No pairs with positive_text. Run export_training_feedback.py first.")

    path_index = build_path_index()
    if args.fold >= 0:
        _, pairs = split_by_path_fold(pairs, args.fold, args.n_folds)

    passages, path_to_idx = build_corpus_from_pairs(pairs, path_index)
    # closed-set: rank full corpus built from all pairs in eval set
    all_passages, all_path_to_idx = build_corpus_from_pairs(
        load_pairs(args.dataset),
        path_index,
    )

    from sentence_transformers import SentenceTransformer

    model_path = str(args.checkpoint) if args.checkpoint else args.model
    model = SentenceTransformer(model_path)

    report = mrr_at_k(
        model,
        pairs,
        all_passages,
        all_path_to_idx,
        k=args.k,
    )
    report["model"] = model_path
    report["fold"] = args.fold
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"model={model_path}")
        print(f"MRR@{report['k']}={report['mrr_at_k']} recall@1={report['recall_at_1']} n={report['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
