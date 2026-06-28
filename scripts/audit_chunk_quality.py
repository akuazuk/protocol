#!/usr/bin/env python3
"""Аудит качества rich-чанков: issues, score, markdown-отчёт."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.chunk_quality import detect_issues, quality_score

DEFAULT_CHUNKS = ROOT / "output" / "rich_chunks" / "rich_chunks.jsonl"
DEFAULT_ISSUES = ROOT / "data" / "ml" / "chunk_qa_issues.jsonl"
DEFAULT_REPORT = ROOT / "data" / "ml" / "reports" / f"chunk_quality_{date.today().isoformat()}.md"
DEFAULT_STATS = ROOT / "data" / "ml" / "reports" / f"chunk_quality_{date.today().isoformat()}.json"


def audit(path: Path, *, limit: int = 0, score_threshold: float = 0.7) -> dict:
    return run_audit(path, limit=limit, score_threshold=score_threshold, issues_out=None)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p)
    return round(s[min(idx, len(s) - 1)], 3)


def run_audit(
    path: Path,
    *,
    limit: int = 0,
    score_threshold: float = 0.7,
    issues_out: Path | None = None,
) -> dict:
    issue_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    indexable_counts: Counter[str] = Counter()
    scores: list[float] = []
    samples: dict[str, list[str]] = {
        "low_score": [],
        "preamble": [],
        "icd_inflation": [],
        "body_clinical": [],
    }
    n = 0
    flagged = 0

    out_fh = issues_out.open("w", encoding="utf-8") if issues_out else None
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if limit and n >= limit:
                    break
                row = json.loads(line)
                n += 1
                ctype = str(row.get("chunk_type") or "body")
                type_counts[ctype] += 1
                score = quality_score(row)
                scores.append(score)
                row["quality_score"] = score
                issues = detect_issues(row)
                for iss in issues:
                    issue_counts[iss] += 1
                idx = row.get("indexable")
                indexable_counts[str(idx)] += 1

                if score < score_threshold or issues:
                    flagged += 1
                    if out_fh:
                        out_fh.write(json.dumps({
                            "chunk_id": row.get("chunk_id"),
                            "doc_id": row.get("doc_id"),
                            "source_path": row.get("source_path"),
                            "chunk_type": ctype,
                            "quality_score": score,
                            "issues": issues,
                            "section_title": row.get("section_title"),
                            "text_preview": (row.get("text") or "")[:200],
                        }, ensure_ascii=False) + "\n")

                cid = str(row.get("chunk_id") or "")
                if score < score_threshold and len(samples["low_score"]) < 5:
                    samples["low_score"].append(cid)
                if "preamble_leak" in issues and len(samples["preamble"]) < 5:
                    samples["preamble"].append(cid)
                if "icd_inflation" in issues and len(samples["icd_inflation"]) < 5:
                    samples["icd_inflation"].append(cid)
                if "type_body_but_clinical" in issues and len(samples["body_clinical"]) < 5:
                    samples["body_clinical"].append(cid)
    finally:
        if out_fh:
            out_fh.close()

    return {
        "chunks_read": n,
        "flagged": flagged,
        "issue_counts": dict(issue_counts.most_common()),
        "type_counts": dict(type_counts.most_common(20)),
        "indexable": dict(indexable_counts),
        "score_avg": round(sum(scores) / max(len(scores), 1), 3),
        "score_p50": _percentile(scores, 0.5),
        "samples": samples,
    }


def write_report(stats: dict, report_path: Path, *, baseline: dict | None = None) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Chunk Quality Report ({date.today().isoformat()})",
        "",
        "## Summary",
        f"- Chunks read: **{stats['chunks_read']}**",
        f"- Flagged (score < threshold or issues): **{stats['flagged']}**",
        f"- Avg quality_score: **{stats['score_avg']}**",
        f"- Median quality_score: **{stats['score_p50']}**",
        "",
        "## Issue counts",
        "",
        "| Issue | Count |",
        "|-------|------:|",
    ]
    for k, v in stats.get("issue_counts", {}).items():
        lines.append(f"| `{k}` | {v} |")
    lines.extend([
        "",
        "## Chunk types (top)",
        "",
        "| Type | Count |",
        "|------|------:|",
    ])
    for k, v in stats.get("type_counts", {}).items():
        lines.append(f"| `{k}` | {v} |")
    lines.extend([
        "",
        "## Indexable",
        "",
    ])
    for k, v in stats.get("indexable", {}).items():
        lines.append(f"- `{k}`: {v}")
    lines.extend([
        "",
        "## Samples",
        "",
    ])
    for label, ids in stats.get("samples", {}).items():
        lines.append(f"- **{label}**: {', '.join(ids) or '-'}")
    if baseline:
        lines.extend([
            "",
            "## vs baseline",
            "",
            f"- Baseline avg score: {baseline.get('score_avg', '?')}",
            f"- Delta avg: {round(stats['score_avg'] - float(baseline.get('score_avg') or 0), 3)}",
        ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit rich chunk quality")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_ISSUES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    if not args.chunks.is_file():
        print(f"Нет файла: {args.chunks}", file=sys.stderr)
        return 1

    baseline = None
    if args.baseline and args.baseline.is_file():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    stats = run_audit(
        args.chunks,
        limit=args.limit,
        score_threshold=args.threshold,
        issues_out=args.out_jsonl,
    )
    write_report(stats, args.report, baseline=baseline)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
