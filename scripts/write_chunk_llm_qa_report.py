#!/usr/bin/env python3
"""Markdown-отчёт по LLM chunk QA fixes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXES = ROOT / "data" / "ml" / "chunk_qa_fixes.jsonl"
DEFAULT_REPORT = ROOT / "data" / "ml" / "reports" / f"chunk_llm_qa_{date.today().isoformat()}.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write LLM chunk QA report")
    parser.add_argument("--fixes", type=Path, default=DEFAULT_FIXES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--merge-stats", type=Path, default=None)
    args = parser.parse_args()

    if not args.fixes.is_file():
        print(f"Нет {args.fixes}", file=__import__("sys").stderr)
        return 1

    verdicts: Counter[str] = Counter()
    conf_sum = 0.0
    n = 0
    samples: dict[str, list[str]] = {"drop": [], "fix": [], "merge_with_next": []}
    for line in args.fixes.open(encoding="utf-8"):
        row = json.loads(line)
        n += 1
        v = str(row.get("verdict") or "ok")
        verdicts[v] += 1
        conf_sum += float(row.get("confidence") or 0)
        if v in samples and len(samples[v]) < 5:
            samples[v].append(str(row.get("chunk_id")))

    merge_stats = {}
    if args.merge_stats and args.merge_stats.is_file():
        merge_stats = json.loads(args.merge_stats.read_text(encoding="utf-8"))

    lines = [
        f"# LLM Chunk QA Report ({date.today().isoformat()})",
        "",
        f"- Fixes total: **{n}**",
        f"- Avg confidence: **{round(conf_sum / max(n, 1), 3)}**",
        "",
        "## Verdicts",
        "",
        "| Verdict | Count |",
        "|---------|------:|",
    ]
    for k, v in verdicts.most_common():
        lines.append(f"| `{k}` | {v} |")
    if merge_stats:
        lines.extend(["", "## Merge apply", "", f"```json\n{json.dumps(merge_stats, ensure_ascii=False, indent=2)}\n```"])
    lines.extend(["", "## Samples", ""])
    for k, ids in samples.items():
        lines.append(f"- **{k}**: {', '.join(ids) or '-'}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
