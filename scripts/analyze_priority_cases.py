#!/usr/bin/env python3
"""Markdown-отчёт по priority_cases после export feedback."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "ml" / "datasets"
PRIORITY = DATASETS / "priority_cases.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def build_report(*, priority_path: Path | None = None) -> str:
    path = priority_path or PRIORITY
    cases = _load_jsonl(path)
    tag_counts: Counter[str] = Counter()
    for c in cases:
        for t in c.get("tags") or []:
            tag_counts[str(t)] += 1

    lines = [
        "# Priority cases triage",
        "",
        f"Источник: `{path}`",
        f"Кейсов (rating≤2): **{len(cases)}**",
        "",
    ]
    if tag_counts:
        lines.append("## Теги")
        for tag, n in tag_counts.most_common():
            lines.append(f"- `{tag}`: {n}")
        lines.append("")

    lines.append("## Кейсы")
    lines.append("")
    lines.append("| hash | rating | verdict | rubric/tags |")
    lines.append("|------|--------|---------|-------------|")
    for c in cases:
        th = str(c.get("text_hash") or "")[:20]
        tags = ", ".join(c.get("tags") or [])[:60]
        lines.append(
            f"| `{th}` | {c.get('rating')} | {c.get('verdict')} | {tags} |"
        )
    lines.append("")
    lines.append("## Рекомендуемые actions")
    lines.append("")
    if tag_counts.get("false_positive_rule"):
        lines.append("- Engine: rule family gates / condition context (см. overrides в analysis_review)")
    if tag_counts.get("wrong_protocol") or tag_counts.get("missed_protocol"):
        lines.append("- RAG: собрать retrieval_fix; проверить rubric pre-filter")
    if tag_counts.get("score_misleading"):
        lines.append("- Compliance: caps sparse KZ, hybrid weights")
    if not cases:
        lines.append("- Нет priority_cases — продолжайте batch + разметку")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--priority", type=Path, default=PRIORITY)
    ap.add_argument("--out", type=Path, default=ROOT / "ml" / "reports" / "priority_triage.md")
    args = ap.parse_args()
    text = build_report(priority_path=args.priority)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
