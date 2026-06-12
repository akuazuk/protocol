#!/usr/bin/env python3
"""Первый эксперимент: baseline e5 vs fine-tune, 5-fold CV по path, golden hold-out.

  python3 scripts/run_embedder_experiment.py
  python3 scripts/run_embedder_experiment.py --quick   # 1 fold, 1 epoch
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ML = ROOT / "ml"
DATASET = ML / "datasets" / "retrieval_pairs_resolved.jsonl"
GOLDEN = ROOT / "eval" / "golden_queries.prod.jsonl"
EXP_DIR = ML / "experiments" / "embedder_exp_001"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    import os

    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    subprocess.run(cmd, cwd=cwd or ROOT, check=True, env=env)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _golden_pairs() -> list[dict]:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from export_training_feedback import load_golden_query_pairs

    return load_golden_query_pairs()


def _eval_model(model_ref: str, test_pairs: list[dict], corpus_pairs: list[dict], k: int = 10) -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ml.eval.eval_embedder_mrr import build_corpus_from_pairs, mrr_at_k
    from ml.chunk_resolver import build_path_index
    from sentence_transformers import SentenceTransformer

    path_index = build_path_index()
    passages, path_to_idx = build_corpus_from_pairs(corpus_pairs, path_index)

    model = SentenceTransformer(model_ref, device="cpu")
    return mrr_at_k(model, test_pairs, passages, path_to_idx, k=k)


def _enrich_golden_with_text(golden_rows: list[dict]) -> list[dict]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ml.chunk_resolver import build_path_index, resolve_path_text

    index = build_path_index()
    out = []
    for row in golden_rows:
        path = (row.get("positive_path") or "").strip()
        text = resolve_path_text(path, index) if path else None
        if path and text:
            out.append({**row, "positive_text": text})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="1 fold, 1 epoch")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1) export + resolve texts
    _run([sys.executable, str(SCRIPTS / "export_training_feedback.py"), "--seed-only"])

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ml.eval.eval_embedder_mrr import load_pairs, path_fold, split_by_path_fold

    all_pairs = load_pairs(DATASET)
    golden_raw = _golden_pairs()
    golden_pairs = _enrich_golden_with_text(golden_raw)

    report: dict = {
        "experiment": "embedder_exp_001",
        "base_model": "intfloat/multilingual-e5-small",
        "dataset": str(DATASET.relative_to(ROOT)),
        "train_pairs": len(all_pairs),
        "golden_holdout": len(golden_pairs),
        "epochs": args.epochs,
        "n_folds": args.n_folds,
        "cv_folds": [],
    }

    # 2) baseline golden (never in train - separate queries)
    print("\n=== Baseline e5-small (golden hold-out) ===", flush=True)
    baseline_golden = _eval_model(report["base_model"], golden_pairs, all_pairs)
    report["baseline_golden"] = baseline_golden
    print(json.dumps(baseline_golden, ensure_ascii=False, indent=2))

    folds = [0] if args.quick else range(args.n_folds)
    n_folds_split = 5
    cv_mrr_before: list[float] = []
    cv_mrr_after: list[float] = []

    for fold in folds:
        train_rows, test_rows = split_by_path_fold(all_pairs, fold, n_folds_split)
        if not test_rows:
            continue
        print(f"\n=== Fold {fold}: test n={len(test_rows)} train n={len(train_rows)} ===", flush=True)

        before = _eval_model(report["base_model"], test_rows, all_pairs)
        cv_mrr_before.append(before["mrr_at_k"])
        print("baseline fold:", before)

        fold_ckpt = EXP_DIR / f"checkpoint_fold_{fold}"
        if not args.skip_train:
            _run(
                [
                    sys.executable,
                    str(ML / "train" / "finetune_embedder.py"),
                    "--dataset",
                    str(DATASET),
                    "--output",
                    str(fold_ckpt),
                    "--test-fold",
                    str(fold),
                    "--n-folds",
                    str(n_folds_split),
                    "--epochs",
                    str(args.epochs),
                    "--device",
                    "cpu",
                    "--batch-size",
                    "4",
                ]
            )
            after = _eval_model(str(fold_ckpt), test_rows, all_pairs)
        else:
            after = before

        cv_mrr_after.append(after["mrr_at_k"])
        report["cv_folds"].append(
            {
                "fold": fold,
                "test_n": len(test_rows),
                "train_n": len(train_rows),
                "baseline": before,
                "finetuned": after,
                "delta_mrr": round(after["mrr_at_k"] - before["mrr_at_k"], 4),
            }
        )
        print("finetuned fold:", after)

    report["cv_summary"] = {
        "baseline_mrr_mean": round(sum(cv_mrr_before) / len(cv_mrr_before), 4) if cv_mrr_before else 0,
        "finetuned_mrr_mean": round(sum(cv_mrr_after) / len(cv_mrr_after), 4) if cv_mrr_after else 0,
    }
    if cv_mrr_before and cv_mrr_after:
        report["cv_summary"]["delta_mrr_mean"] = round(
            report["cv_summary"]["finetuned_mrr_mean"] - report["cv_summary"]["baseline_mrr_mean"],
            4,
        )

    # final model on all data
    final_ckpt = EXP_DIR / "checkpoint_final"
    if not args.skip_train:
        _run(
            [
                sys.executable,
                str(ML / "train" / "finetune_embedder.py"),
                "--dataset",
                str(DATASET),
                "--output",
                str(final_ckpt),
                "--test-fold",
                "-1",
                "--n-folds",
                "5",
                "--epochs",
                str(args.epochs),
                "--device",
                "cpu",
                "--batch-size",
                "4",
            ]
        )
        # train with all rows when test_fold=-1 - need to fix finetune for that
        report["final_checkpoint"] = str(final_ckpt.relative_to(ROOT))

    report["elapsed_sec"] = round(time.time() - t0, 1)
    if not args.skip_train and (EXP_DIR / "checkpoint_final").is_dir():
        print("\n=== Fine-tuned final (golden hold-out) ===", flush=True)
        report["finetuned_golden"] = _eval_model(
            str(EXP_DIR / "checkpoint_final"), golden_pairs, all_pairs
        )
        print(json.dumps(report["finetuned_golden"], ensure_ascii=False, indent=2))

    report_path = EXP_DIR / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = _format_report_md(report)
    (EXP_DIR / "REPORT.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"\nWrote {report_path}")
    return 0


def _format_report_md(r: dict) -> str:
    lines = [
        "# Embedder experiment 001",
        "",
        f"- Base model: `{r['base_model']}`",
        f"- Train pairs (resolved): **{r['train_pairs']}**",
        f"- Golden hold-out queries: **{r['golden_holdout']}**",
        f"- Epochs: {r['epochs']}",
        "",
        "## Golden hold-out (baseline e5-small)",
        "",
        f"- MRR@10: **{r.get('baseline_golden', {}).get('mrr_at_k', 'n/a')}**",
        f"- Recall@1: **{r.get('baseline_golden', {}).get('recall_at_1', 'n/a')}**",
        "",
    ]
    if r.get("finetuned_golden"):
        fg = r["finetuned_golden"]
        lines.extend(
            [
                "## Golden hold-out (fine-tuned final)",
                "",
                f"- MRR@10: **{fg.get('mrr_at_k', 'n/a')}**",
                f"- Recall@1: **{fg.get('recall_at_1', 'n/a')}**",
                "",
            ]
        )
    lines.extend(
        [
        "## 5-fold CV (by protocol path)",
        "",
    ]
    )
    cv = r.get("cv_summary") or {}
    lines.append(f"- Baseline MRR mean: **{cv.get('baseline_mrr_mean', 'n/a')}**")
    lines.append(f"- Fine-tuned MRR mean: **{cv.get('finetuned_mrr_mean', 'n/a')}**")
    lines.append(f"- Delta: **{cv.get('delta_mrr_mean', 'n/a')}**")
    lines.append("")
    for fold in r.get("cv_folds") or []:
        lines.append(
            f"- Fold {fold['fold']}: baseline MRR={fold['baseline']['mrr_at_k']} "
            f"→ finetuned={fold['finetuned']['mrr_at_k']} (Δ={fold['delta_mrr']})"
        )
    lines.append("")
    lines.append(f"Elapsed: {r.get('elapsed_sec')}s")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
