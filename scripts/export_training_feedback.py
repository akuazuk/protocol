#!/usr/bin/env python3
"""Экспорт обезличенных датасетов для fine-tune из feedback и golden sets.

Читает:
  - data/ml/feedback/*.jsonl  (события пилота)
  - data/quality_benchmark.json (пары запрос-протокол для RAG)
  - data/gastro_mvp/consult_gold.jsonl (регрессия KZ)

Пишет в ml/datasets/:
  - retrieval_pairs.jsonl
  - entailment_pairs.jsonl
  - kz_regression.jsonl
  - export_manifest.json

Пример:
  python3 scripts/export_training_feedback.py
  python3 scripts/export_training_feedback.py --seed-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEEDBACK_DIR = ROOT / "data" / "ml" / "feedback"
DATASETS_DIR = ROOT / "ml" / "datasets"
QUALITY_BENCH = ROOT / "data" / "quality_benchmark.json"
GOLDEN_QUERIES = ROOT / "eval" / "golden_queries.prod.jsonl"
STRUCTURED_INDEX = ROOT / "structured_index.json"
GASTRO_GOLD = ROOT / "data" / "gastro_mvp" / "consult_gold.jsonl"
CONFIG_PATH = ROOT / "ml" / "configs" / "default.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def iter_feedback_events() -> Iterator[dict[str, Any]]:
    if not FEEDBACK_DIR.is_dir():
        return
    for path in sorted(FEEDBACK_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                row["_source_file"] = path.name
                yield row


def load_golden_query_pairs() -> list[dict[str, Any]]:
    """Пары query → path по must_substrings из eval/golden_queries.prod.jsonl."""
    if not GOLDEN_QUERIES.is_file() or not STRUCTURED_INDEX.is_file():
        return []
    structured = json.loads(STRUCTURED_INDEX.read_text(encoding="utf-8"))
    pairs: list[dict[str, Any]] = []
    for line in GOLDEN_QUERIES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("expect_empty"):
            continue
        query = (item.get("query") or "").strip()
        subs = [s.lower() for s in (item.get("must_substrings") or []) if s]
        if not query or not subs:
            continue
        best_path = ""
        best_hits = 0
        for row in structured:
            blob = " ".join(
                str(row.get(k) or "")
                for k in ("title", "search_text", "diagnosis", "path")
            ).lower()
            hits = sum(1 for s in subs if s in blob)
            if hits > best_hits:
                best_hits = hits
                best_path = str(row.get("path") or "")
        if best_path and best_hits >= max(1, len(subs) // 2):
            pairs.append(
                {
                    "query": query,
                    "positive_path": best_path,
                    "source": "golden_queries",
                    "label": 1,
                    "match_hits": best_hits,
                }
            )
    return pairs


def load_quality_retrieval_pairs() -> list[dict[str, Any]]:
    return load_golden_query_pairs()


def load_gastro_gold() -> list[dict[str, Any]]:
    if not GASTRO_GOLD.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in GASTRO_GOLD.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def events_to_retrieval_pairs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("event_type") != "retrieval_fix":
            continue
        query = (ev.get("query") or "").strip()
        chosen = (ev.get("chosen_path") or "").strip()
        rejected = (ev.get("rejected_path") or "").strip()
        if query and chosen:
            out.append(
                {
                    "query": query,
                    "positive_path": chosen,
                    "negative_path": rejected or None,
                    "source": "feedback",
                    "label": 1,
                }
            )
    return out


def events_to_entailment_pairs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in events:
        et = ev.get("event_type")
        if et == "methodist_override":
            rule_id = (ev.get("rule_id") or "").strip()
            note = (ev.get("note") or rule_id).strip()
            text_hash = ev.get("text_hash") or ""
            human_pass = ev.get("human_pass")
            if rule_id and text_hash:
                out.append(
                    {
                        "text_hash": text_hash,
                        "term": rule_id,
                        "label": "entailment" if human_pass else "contradiction",
                        "note": note,
                        "source": "methodist_override",
                    }
                )
        elif et == "l0_screen":
            # placeholder for future: block-level labels without raw text
            pass
    return out


def bootstrap_structured_index_pairs(limit: int = 300) -> list[dict[str, Any]]:
    """Bootstrap: title/diagnosis → path из structured_index.json."""
    if not STRUCTURED_INDEX.is_file():
        return []
    rows = json.loads(STRUCTURED_INDEX.read_text(encoding="utf-8"))
    pairs: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if i >= limit:
            break
        path = str(row.get("path") or "").strip()
        title = str(row.get("title") or "").strip()
        diag = str(row.get("diagnosis") or "").strip()[:280]
        if not path:
            continue
        query = title or diag[:120]
        if not query:
            continue
        pairs.append(
            {
                "query": query,
                "positive_path": path,
                "source": "structured_index_bootstrap",
                "label": 1,
            }
        )
    return pairs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def enrich_retrieval_with_texts(
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Добавляет positive_text / negative_text через ml.chunk_resolver."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ml.chunk_resolver import build_path_index, resolve_path_text

    index = build_path_index()
    stats = {"total": len(pairs), "resolved": 0, "missing": 0, "with_negative": 0}
    out: list[dict[str, Any]] = []
    for row in pairs:
        path = (row.get("positive_path") or "").strip()
        neg_path = (row.get("negative_path") or "").strip()
        pos_text = resolve_path_text(path, index) if path else None
        if not pos_text:
            stats["missing"] += 1
            continue
        stats["resolved"] += 1
        item = dict(row)
        item["positive_text"] = pos_text
        if neg_path:
            neg_text = resolve_path_text(neg_path, index)
            if neg_text:
                item["negative_text"] = neg_text
                stats["with_negative"] += 1
        out.append(item)
    return out, stats


def export_all(*, seed_only: bool = False) -> dict[str, Any]:
    events = [] if seed_only else list(iter_feedback_events())

    retrieval = load_quality_retrieval_pairs()
    retrieval += bootstrap_structured_index_pairs()
    if not seed_only:
        retrieval += events_to_retrieval_pairs(events)

    entailment = events_to_entailment_pairs(events)
    kz_regression = load_gastro_gold()

    # dedupe retrieval by (query, positive_path)
    seen: set[tuple[str, str]] = set()
    retrieval_deduped: list[dict[str, Any]] = []
    for row in retrieval:
        key = (row["query"], row["positive_path"])
        if key in seen:
            continue
        seen.add(key)
        retrieval_deduped.append(row)

    write_jsonl(DATASETS_DIR / "retrieval_pairs.jsonl", retrieval_deduped)
    resolved, resolve_stats = enrich_retrieval_with_texts(retrieval_deduped)
    write_jsonl(DATASETS_DIR / "retrieval_pairs_resolved.jsonl", resolved)
    write_jsonl(DATASETS_DIR / "entailment_pairs.jsonl", entailment)
    write_jsonl(DATASETS_DIR / "kz_regression.jsonl", kz_regression)

    manifest = {
        "exported_at": _utc_now(),
        "seed_only": seed_only,
        "counts": {
            "feedback_events": len(events),
            "retrieval_pairs": len(retrieval_deduped),
            "retrieval_pairs_resolved": len(resolved),
            "resolve_stats": resolve_stats,
            "entailment_pairs": len(entailment),
            "kz_regression": len(kz_regression),
        },
        "sources": {
            "feedback_dir": str(FEEDBACK_DIR.relative_to(ROOT)),
            "golden_queries": GOLDEN_QUERIES.is_file(),
            "structured_index": STRUCTURED_INDEX.is_file(),
            "gastro_gold": GASTRO_GOLD.is_file(),
        },
        "outputs": [
            "ml/datasets/retrieval_pairs.jsonl",
            "ml/datasets/retrieval_pairs_resolved.jsonl",
            "ml/datasets/entailment_pairs.jsonl",
            "ml/datasets/kz_regression.jsonl",
        ],
        "next_steps": [
            "python3 scripts/run_embedder_experiment.py",
            "ml/train/finetune_embedder.py --dataset ml/datasets/retrieval_pairs_resolved.jsonl",
        ],
    }
    manifest_path = DATASETS_DIR / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ML training datasets from feedback")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only bootstrap from quality_benchmark, index.csv, gastro gold (no feedback/)",
    )
    args = parser.parse_args()
    manifest = export_all(seed_only=args.seed_only)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
