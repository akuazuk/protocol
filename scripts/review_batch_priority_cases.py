#!/usr/bin/env python3
"""AI-review для приоритетных кейсов из batch report (предразметка для методиста)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import env_load

    env_load.load_project_env(ROOT)
except ImportError:
    pass

DEFAULT_PRIORITY = (
    "report_n_1",
    "report_n_2",
    "kard_1",
    "gastro_1",
    "F_1_p",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--report",
        type=Path,
        default=ROOT / "ml" / "experiments" / "batch_clients_consult_2026-06-01" / "report.json",
    )
    ap.add_argument("--cases", nargs="*", default=list(DEFAULT_PRIORITY))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.report.is_file():
        print(f"Report not found: {args.report}", file=sys.stderr)
        return 1

    data = json.loads(args.report.read_text(encoding="utf-8"))
    by_case = {r["case_id"]: r for r in (data.get("reports") or []) if r.get("case_id")}
    out_dir = args.out or args.report.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    from clinical_knowledge.feedback_store import load_analysis_snapshot, load_secure_kz_text
    from clinical_knowledge.methodist_ai_review import methodist_ai_review_enabled, run_methodist_ai_review

    if not methodist_ai_review_enabled():
        print("METHODIST_AI_REVIEW отключён", file=sys.stderr)
        return 1

    results: list[dict] = []
    queue_lines = [
        "# Очередь разметки (batch → кабинет методиста)",
        "",
        f"Источник batch: `{args.report.parent.name}`",
        "",
        "В UI: **Очередь** → **Открыть** по `analysis_id`, или вставьте hash в «Проверка КЗ».",
        "",
        "| case_id | analysis_id | overall % | AI готов |",
        "|---------|-------------|-----------|----------|",
    ]

    for case_id in args.cases:
        rep = by_case.get(case_id)
        if not rep:
            print(f"SKIP missing case: {case_id}", file=sys.stderr)
            continue
        aid = str(rep.get("analysis_id") or "")
        snap = load_analysis_snapshot(aid)
        if not snap:
            print(f"SKIP no snapshot: {case_id} {aid}", file=sys.stderr)
            continue
        api_result = snap.get("api_result") or {}
        th = str(snap.get("text_hash") or "")
        text = load_secure_kz_text(th) or str(snap.get("text_excerpt") or "")
        t0 = time.perf_counter()
        try:
            ai = run_methodist_ai_review(api_result, text)
            err = None
        except Exception as exc:
            ai = None
            err = str(exc)[:400]
        ms = int((time.perf_counter() - t0) * 1000)
        row = {
            "case_id": case_id,
            "analysis_id": aid,
            "text_hash": th,
            "overall_pct": rep.get("overall_pct"),
            "failed_rule_ids": rep.get("failed_rule_ids") or [],
            "ai_review_ms": ms,
            "ai_review": ai,
            "ai_error": err,
        }
        results.append(row)
        ok = "✅" if ai and not err else "❌"
        queue_lines.append(
            f"| {case_id} | `{aid[:8]}…` | {rep.get('overall_pct', '—')}% | {ok} |"
        )
        print(f"{'OK' if ai else 'ERR'} {case_id} ({ms} ms)")

    ai_path = out_dir / "priority_ai_reviews.json"
    ai_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    queue_path = out_dir / "REVIEW_QUEUE.md"
    queue_lines.extend(
        [
            "",
            "## Детали",
            "",
        ]
    )
    for row in results:
        queue_lines.append(f"### {row['case_id']}")
        queue_lines.append(f"- `analysis_id`: `{row['analysis_id']}`")
        queue_lines.append(f"- `text_hash`: `{row.get('text_hash', '')[:24]}…`")
        if row.get("ai_error"):
            queue_lines.append(f"- AI error: {row['ai_error']}")
        elif row.get("ai_review"):
            ai = row["ai_review"]
            queue_lines.append(
                f"- AI rating: **{ai.get('system_accuracy_rating')}** · "
                f"verdict: `{ai.get('system_accuracy_verdict')}`"
            )
            if ai.get("summary_ru"):
                queue_lines.append(f"- {ai['summary_ru'][:300]}")
        queue_lines.append("")

    queue_path.write_text("\n".join(queue_lines), encoding="utf-8")
    print(f"\nSaved: {ai_path}")
    print(f"Saved: {queue_path}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
