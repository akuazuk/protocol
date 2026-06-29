#!/usr/bin/env python3
"""Train chunk issue classifier v1 (TF-IDF + multi-label logistic regression).

Gate (action-plan): F1 >= 0.85 on preamble_leak when n_val >= 30.

Example:
  python3 scripts/export_chunk_qa_dataset.py
  python3 ml/train/train_chunk_classifier.py
  python3 ml/train/train_chunk_classifier.py --dataset ml/datasets/chunk_qa_classifier.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATASET = ROOT / "ml/datasets/chunk_qa_classifier.jsonl"
DEFAULT_OUT = ROOT / "ml/registry/checkpoints/chunk_classifier_v1"
REPORT_DIR = ROOT / "ml/experiments/chunk_classifier_v1"

P0_ISSUES = ("preamble_leak", "icd_inflation")
GATE_F1 = 0.85
MIN_P0_VAL = 30
TOP_ISSUES = (
    "weak_section_title",
    "too_long",
    "truncated_list",
    "too_short",
    "type_body_but_clinical",
    "empty_entities",
    "preamble_leak",
    "icd_inflation",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def split_by_doc(rows: list[dict[str, Any]], *, test_ratio: float = 0.2, seed: int = 42) -> tuple[list, list]:
    import random

    by_doc: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_doc.setdefault(str(r.get("doc_id") or "unknown"), []).append(r)
    docs = sorted(by_doc)
    random.seed(seed)
    random.shuffle(docs)
    n_test = max(1, int(len(docs) * test_ratio))
    test_docs = set(docs[:n_test])
    train, test = [], []
    for doc, items in by_doc.items():
        (test if doc in test_docs else train).extend(items)
    return train, test


def _build_text(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("chunk_type") or ""),
        str(row.get("section_title") or ""),
        str(row.get("text") or ""),
    ]
    return " ".join(p for p in parts if p).strip()


def train_and_eval(
    *,
    dataset: Path,
    output: Path,
    report_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    try:
        import joblib
        import numpy as np
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            precision_recall_fscore_support,
        )
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer
    except ImportError as e:
        raise SystemExit(
            "Install ML deps: pip install scikit-learn joblib numpy\n" + str(e)
        ) from e

    rows = load_rows(dataset)
    if len(rows) < 100:
        raise SystemExit(f"Too few rows in {dataset}: {len(rows)}")

    train_rows, test_rows = split_by_doc(rows, seed=seed)
    issue_vocab = sorted({i for r in rows for i in (r.get("issues") or []) if i})
    if not issue_vocab:
        raise SystemExit("No issue labels in dataset")

    train_texts = [_build_text(r) for r in train_rows]
    test_texts = [_build_text(r) for r in test_rows]

    vec = HashingVectorizer(n_features=2**18, alternate_sign=False, ngram_range=(1, 2))
    x_train = vec.transform(train_texts)
    x_test = vec.transform(test_texts)

    mlb = MultiLabelBinarizer(classes=issue_vocab)
    y_train = mlb.fit_transform([r.get("issues") or [] for r in train_rows])
    y_test = mlb.transform([r.get("issues") or [] for r in test_rows])

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed),
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    clf.fit(x_train, y_train)
    train_sec = time.perf_counter() - t0

    y_pred = clf.predict(x_test)
    f1_micro = float(f1_score(y_test, y_pred, average="micro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    per_issue: dict[str, dict[str, float | int]] = {}
    for idx, label in enumerate(mlb.classes_):
        p, r, f1, support = precision_recall_fscore_support(
            y_test[:, idx], y_pred[:, idx], average="binary", zero_division=0
        )
        per_issue[label] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": int((y_test[:, idx] == 1).sum()),
        }

    # needs_action (verdict / any issue)
    y_na_train = np.array([int(r.get("needs_action") or 0) for r in train_rows])
    y_na_test = np.array([int(r.get("needs_action") or 0) for r in test_rows])
    na_clf = LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed)
    na_clf.fit(x_train, y_na_train)
    y_na_pred = na_clf.predict(x_test)
    na_acc = float(accuracy_score(y_na_test, y_na_pred))
    na_f1 = float(f1_score(y_na_test, y_na_pred, average="binary", zero_division=0))

    p0_gate: dict[str, Any] = {}
    gate_pass = True
    for label in P0_ISSUES:
        stats = per_issue.get(label) or {"f1": 0.0, "support": 0}
        sup = int(stats.get("support") or 0)
        f1 = float(stats.get("f1") or 0.0)
        if sup >= MIN_P0_VAL:
            ok = f1 >= GATE_F1
            p0_gate[label] = {"f1": f1, "support": sup, "gate_f1": GATE_F1, "pass": ok}
            if not ok:
                gate_pass = False
        else:
            p0_gate[label] = {
                "f1": f1,
                "support": sup,
                "gate_f1": GATE_F1,
                "pass": None,
                "note": f"skip gate (need >={MIN_P0_VAL} val positives)",
            }

    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "vectorizer": vec,
        "issue_clf": clf,
        "label_binarizer": mlb,
        "needs_action_clf": na_clf,
        "issue_vocab": issue_vocab,
    }
    joblib.dump(bundle, output / "model.joblib")

    meta = {
        "dataset": str(dataset),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "issue_labels": len(issue_vocab),
        "train_seconds": round(train_sec, 2),
        "metrics": {
            "issues_f1_micro": round(f1_micro, 4),
            "issues_f1_macro": round(f1_macro, 4),
            "needs_action_accuracy": round(na_acc, 4),
            "needs_action_f1": round(na_f1, 4),
        },
        "per_issue": {k: per_issue[k] for k in TOP_ISSUES if k in per_issue},
        "p0_gate": p0_gate,
        "gate_pass": gate_pass,
        "model_path": str(output / "model.joblib"),
    }
    (output / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    report_md = [
        "# Chunk classifier v1",
        "",
        f"- Dataset: `{dataset}`",
        f"- Train/test: {len(train_rows)} / {len(test_rows)} (split by doc_id)",
        f"- Train time: **{train_sec:.1f}s** (CPU, HashingVectorizer + OvR LogReg)",
        "",
        "## Metrics",
        "",
        f"- Issues F1 micro: **{f1_micro:.3f}**",
        f"- Issues F1 macro: **{f1_macro:.3f}**",
        f"- needs_action F1: **{na_f1:.3f}**",
        "",
        "## P0 gate (F1 >= {:.2f})".format(GATE_F1),
        "",
    ]
    for label, g in p0_gate.items():
        status = "PASS" if g.get("pass") is True else ("SKIP" if g.get("pass") is None else "FAIL")
        report_md.append(f"- `{label}`: F1={g.get('f1', 0):.3f} support={g.get('support')} → **{status}**")
    report_md.append("")
    report_md.append(f"**Overall gate:** {'PASS' if gate_pass else 'FAIL (do not enable skip-Gemini yet)'}")
    (report_dir / "REPORT.md").write_text("\n".join(report_md) + "\n", encoding="utf-8")
    (report_dir / "report.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Train chunk QA classifier v1")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dataset.is_file():
        raise SystemExit(
            f"Missing {args.dataset}. Run: python3 scripts/export_chunk_qa_dataset.py"
        )

    meta = train_and_eval(
        dataset=args.dataset,
        output=args.output,
        report_dir=args.report_dir,
        seed=args.seed,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0 if meta.get("gate_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
