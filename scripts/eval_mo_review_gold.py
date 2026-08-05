#!/usr/bin/env python3
"""Сводка согласованности методиста vs LLM/findings по export gold.

Пример:
  python3 scripts/eval_mo_review_gold.py --gold-dir data/medical_exams/gold_review/2026-08-05
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    packs = _load_jsonl(args.gold_dir / "review_packs.jsonl")
    ratings = _load_jsonl(args.gold_dir / "protocol_ratings.jsonl")
    verdicts = Counter()
    finding_labels = Counter()
    for pack in packs:
        decision = pack.get("decision") or {}
        for key in ("verdict_completeness", "verdict_diagnosis", "verdict_recommendations"):
            verdicts[f"{key}:{decision.get(key) or 'unreviewed'}"] += 1
        for code, label in (decision.get("finding_decisions") or {}).items():
            finding_labels[str(label)] += 1
    relevance = Counter(str(r.get("relevance") or "unreviewed") for r in ratings)
    report = {
        "packs": len(packs),
        "protocol_ratings": len(ratings),
        "verdicts": dict(verdicts),
        "finding_labels": dict(finding_labels),
        "protocol_relevance": dict(relevance),
        "notes": [
            "Disagree/partial по вердиктам - кандидаты на калибровку LLM-judge.",
            "false_positive по findings - кандидаты на ослабление rules.",
            "irrelevant по протоколам - hard negatives для suggest.",
        ],
    }
    text = "# MO review gold eval\n\n```json\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n"
    out = args.out or (args.gold_dir / "REPORT.md")
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
