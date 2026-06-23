#!/usr/bin/env python3
"""Импорт результатов пилота КЗ (render_reviews.json) в ML feedback для export_training_feedback.

Пример:
  python3 scripts/import_kz_pilot_feedback.py
  python3 scripts/import_kz_pilot_feedback.py --experiment ml/experiments/kz_pilot_2026-06-18
  python3 scripts/export_training_feedback.py
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.feedback_store import feedback_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_reviews(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("results") or [])
    if isinstance(data, list):
        return data
    return []


def _to_events(rows: list[dict], *, source_file: str) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        rating = row.get("rating")
        try:
            rating_n = int(rating)
        except (TypeError, ValueError):
            continue
        if rating_n > 2:
            continue
        out.append(
            {
                "event_id": row.get("event_id") or str(uuid.uuid4()),
                "event_type": "analysis_review",
                "ts": _utc_now(),
                "analysis_id": row.get("analysis_id"),
                "case_id": row.get("case_id"),
                "text_hash": row.get("text_hash"),
                "rating": rating_n,
                "verdict": row.get("verdict"),
                "tags": row.get("tags") or [],
                "overall_pct": row.get("overall_pct"),
                "latency_ms": row.get("ms"),
                "source": "kz_pilot_import",
                "source_file": source_file,
            }
        )
    return out


def import_experiment(exp_dir: Path, *, dry_run: bool = False) -> dict:
    reviews_path = exp_dir / "render_reviews.json"
    if not reviews_path.is_file():
        raise FileNotFoundError(f"Нет {reviews_path}")
    rows = _load_reviews(reviews_path)
    events = _to_events(rows, source_file=str(reviews_path.relative_to(ROOT)))
    fb = feedback_dir()
    out_path = fb / "events.jsonl"
    existing_hashes: set[str] = set()
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            th = str(ev.get("text_hash") or "")
            aid = str(ev.get("analysis_id") or "")
            if th and aid:
                existing_hashes.add(f"{th}|{aid}")
    appended = 0
    if not dry_run:
        fb.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            for ev in events:
                key = f"{ev.get('text_hash')}|{ev.get('analysis_id')}"
                if key in existing_hashes:
                    continue
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                existing_hashes.add(key)
                appended += 1
    return {
        "experiment": str(exp_dir),
        "reviews_total": len(rows),
        "priority_events": len(events),
        "appended": appended if not dry_run else 0,
        "feedback_file": str(out_path),
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import KZ pilot render reviews into ML feedback")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=ROOT / "ml" / "experiments" / "kz_pilot_2026-06-18",
        help="Папка эксперимента с render_reviews.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = import_experiment(args.experiment.resolve(), dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.dry_run and summary.get("appended"):
        print("\nДалее: python3 scripts/export_training_feedback.py")


if __name__ == "__main__":
    main()
