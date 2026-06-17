#!/usr/bin/env python3
"""Batch-прогон probe-запросов поиска протоколов с AI-оценкой (как режим методиста).

Использует воронку шаг 4 (hybrid ICD+RAG) + build_deterministic_search_ai_review.

Пример:
  python scripts/run_methodist_search_probe.py
  python scripts/run_methodist_search_probe.py --limit 5 --out data/ml/reports/probe_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "search_methodist_probe.jsonl"
DEFAULT_OUT = ROOT / "data" / "ml" / "reports" / "methodist_search_probe_latest.jsonl"
DEFAULT_MD = ROOT / "data" / "ml" / "reports" / "methodist_search_probe_latest.md"
DEFAULT_SNAPSHOT = ROOT / "data" / "ml" / "search_probe_snapshot.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def _ensure_rag_loaded() -> None:
    import rag_server as rs

    if rs._chunks_load_done.is_set():
        rs._require_rag_loaded()
        return
    print("Загрузка индекса RAG (один раз)…", flush=True)
    t0 = time.perf_counter()
    rs._run_load_data_background()
    rs._require_rag_loaded()
    print(f"RAG готов за {time.perf_counter() - t0:.1f}s", flush=True)


def _run_probe(row: dict[str, Any]) -> dict[str, Any]:
    from clinical_knowledge.methodist_search_probe_runner import run_single_probe

    return run_single_probe(row)


def _summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    from clinical_knowledge.methodist_search_probe_runner import summarize_probe_reports

    return summarize_probe_reports(reports)


def _markdown(summary: dict[str, Any], fixture: Path, version: str) -> str:
    lines = [
        "# Methodist search probe report",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Fixture: `{fixture}`",
        f"- BUILD: `{version}`",
        f"- Probes: **{summary['n_total']}** (ok {summary['n_ok']}, errors {summary['n_error']})",
        "",
    ]
    if summary.get("avg_ai_rating") is not None:
        lines.append(f"- Avg AI rating (deterministic): **{summary['avg_ai_rating']}** / 5")
    if summary.get("expected_hit1_pct") is not None:
        lines.append(
            f"- Expected fragment in top-1: **{summary['expected_hit1_pct']:.1%}** · top-3: **{summary['expected_hit3_pct']:.1%}**"
        )
    lines.extend(
        [
            f"- Top-1 clinically irrelevant (AI): **{summary['top1_not_relevant_count']}**",
            f"- Reject fragment in top-1: **{summary['reject_in_top1_count']}**",
            "",
            "## Verdicts",
            "",
        ]
    )
    for v, n in sorted((summary.get("verdict_counts") or {}).items(), key=lambda x: -x[1]):
        lines.append(f"- `{v}`: {n}")
    if summary.get("tag_counts"):
        lines.extend(["", "## Tags", ""])
        for t, n in summary["tag_counts"].items():
            lines.append(f"- `{t}`: {n}")
    if summary.get("engine_improvements_top"):
        lines.extend(["", "## Recurring engine improvements", ""])
        for imp, n in summary["engine_improvements_top"]:
            lines.append(f"- ({n}×) {imp}")
    lines.extend(["", "## Worst cases", ""])
    lines.append("| id | rating | verdict | top-1 | hit1 | reject |")
    lines.append("|----|--------|---------|-------|------|--------|")
    for r in summary.get("worst") or []:
        lines.append(
            f"| {r.get('id')} | {r.get('ai_rating')} | {r.get('ai_verdict')} | "
            f"{(r.get('top1_short') or '')[:48]} | {r.get('expected_hit1')} | {r.get('reject_in_top1')} |"
        )
    if summary.get("errors"):
        lines.extend(["", "## Errors", ""])
        for e in summary["errors"]:
            lines.append(f"- **{e.get('id')}**: {e.get('error')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Methodist-style batch search probe")
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    _ensure_rag_loaded()

    rows = _load_jsonl(args.fixture)
    if args.limit > 0:
        rows = rows[: args.limit]

    reports: list[dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        rid = row.get("id") or f"row{i}"
        print(f"[{i}/{len(rows)}] {rid} …", flush=True)
        reports.append(_run_probe(row))

    from rag_server import BUILD_VERSION

    summary = _summarize(reports)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["fixture"] = str(args.fixture)
    summary["build_version"] = BUILD_VERSION

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps({"_summary": summary}, ensure_ascii=False) + "\n")

    md = _markdown(summary, args.fixture, BUILD_VERSION)
    args.md.write_text(md, encoding="utf-8")

    dated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dated_jsonl = args.out.parent / f"methodist_search_probe_{dated}.jsonl"
    dated_md = args.md.parent / f"methodist_search_probe_{dated}.md"
    dated_jsonl.write_text(args.out.read_text(encoding="utf-8"), encoding="utf-8")
    dated_md.write_text(md, encoding="utf-8")

    snapshot = {
        "kind": "methodist_search_probe",
        "generated_at": summary["generated_at"],
        "build_version": BUILD_VERSION,
        "fixture": str(args.fixture.relative_to(ROOT)) if args.fixture.is_relative_to(ROOT) else str(args.fixture),
        "n_total": summary["n_total"],
        "n_ok": summary["n_ok"],
        "n_error": summary["n_error"],
        "avg_ai_rating": summary.get("avg_ai_rating"),
        "expected_hit1_pct": summary.get("expected_hit1_pct"),
        "expected_hit3_pct": summary.get("expected_hit3_pct"),
        "verdict_counts": summary.get("verdict_counts"),
        "tag_counts": summary.get("tag_counts"),
        "group_counts": summary.get("group_counts"),
        "reject_in_top1_count": summary.get("reject_in_top1_count"),
        "top1_not_relevant_count": summary.get("top1_not_relevant_count"),
        "worst_ids": [r.get("id") for r in (summary.get("worst") or [])[:5] if r.get("id")],
        "report_jsonl": str(args.out.relative_to(ROOT)) if args.out.is_relative_to(ROOT) else str(args.out),
        "report_md": str(args.md.relative_to(ROOT)) if args.md.is_relative_to(ROOT) else str(args.md),
    }
    DEFAULT_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print()
    print(md)
    print(f"JSONL: {args.out}")
    print(f"Markdown: {args.md}")
    print(f"Dated: {dated_jsonl}")
    print(f"Snapshot: {DEFAULT_SNAPSHOT}")
    return 0 if not summary.get("n_error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
