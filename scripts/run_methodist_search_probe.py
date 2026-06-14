#!/usr/bin/env python3
"""Batch-прогон probe-запросов поиска протоколов с AI-оценкой (как режим методиста).

Использует retrieve_only (/api/assist) + build_deterministic_search_ai_review
(без Gemini; при наличии ключа можно добавить --llm позже).

Пример:
  python scripts/run_methodist_search_probe.py
  python scripts/run_methodist_search_probe.py --limit 5 --out data/ml/reports/probe_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "search_methodist_probe.jsonl"
DEFAULT_OUT = ROOT / "data" / "ml" / "reports" / "methodist_search_probe_latest.jsonl"
DEFAULT_MD = ROOT / "data" / "ml" / "reports" / "methodist_search_probe_latest.md"

_POPULATION_LINE = {
    "adult": "Контекст подбора: взрослое население",
    "pediatric": "Контекст подбора: детское население",
    "pregnant": "Контекст подбора: беременные",
    "emergency": "Контекст подбора: неотложная помощь",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def _basename(path: str) -> str:
    p = (path or "").replace("\\", "/").strip()
    return p.rsplit("/", 1)[-1] if p else ""


def _build_query(row: dict[str, Any]) -> str:
    parts = [str(row.get("query") or "").strip()]
    pop = str(row.get("population") or "").strip().lower()
    if pop in _POPULATION_LINE:
        parts.append(_POPULATION_LINE[pop])
    icd = list(row.get("icd_codes") or [])
    if icd:
        parts.append("МКБ-10: " + ", ".join(str(c) for c in icd))
    return "\n".join(p for p in parts if p)


def _path_matches(path: str, fragments: list[str]) -> bool:
    pl = (path or "").lower()
    return any(str(f).lower() in pl for f in fragments if f)


def _top_paths(data: dict[str, Any], n: int = 5) -> list[str]:
    protos = (data.get("llm_json") or {}).get("protocols") or []
    out: list[str] = []
    for pr in protos[:n]:
        if isinstance(pr, dict) and pr.get("path"):
            out.append(str(pr["path"]))
    return out


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
    from clinical_knowledge.methodist_search_ai_review import build_deterministic_search_ai_review
    from fastapi import HTTPException
    from rag_server import AssistIn, api_assist

    q = _build_query(row)
    slugs = [s for s in (row.get("category_slugs") or []) if isinstance(s, str)]
    t0 = time.perf_counter()
    try:
        data = api_assist(AssistIn(query=q, category_slugs=slugs, retrieve_only=True))
    except HTTPException as exc:
        detail = exc.detail
        if not isinstance(detail, str):
            detail = str(detail)
        return {
            "id": row.get("id"),
            "group": row.get("group"),
            "query": q,
            "error": detail,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
    except Exception as exc:
        return {
            "id": row.get("id"),
            "group": row.get("group"),
            "query": q,
            "error": str(exc),
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
    latency_ms = int((time.perf_counter() - t0) * 1000)
    top_paths = _top_paths(data)
    top1 = top_paths[0] if top_paths else ""
    protos = (data.get("llm_json") or {}).get("protocols") or []
    top_conf = None
    if protos and isinstance(protos[0], dict):
        try:
            top_conf = round(float(protos[0].get("confidence_score") or 0) * 100, 1)
        except (TypeError, ValueError):
            top_conf = None

    icd_codes = list(row.get("icd_codes") or [])
    for bucket in ("detected", "suggested"):
        for r in (data.get("icd") or {}).get(bucket) or []:
            if isinstance(r, dict) and r.get("code"):
                c = str(r["code"])
                if c not in icd_codes:
                    icd_codes.append(c)

    ai = build_deterministic_search_ai_review(
        {
            "query": q,
            "llm_json": data.get("llm_json") or {},
            "retrieval": data.get("retrieval") or [],
            "icd_codes": icd_codes,
            "retrieve_only": True,
            "funnel_context": {"population": row.get("population")},
            "audience_inferred": data.get("audience_inferred"),
        }
    )

    expected = list(row.get("expected_contains") or [])
    reject = list(row.get("reject_contains") or [])
    hit1 = _path_matches(top1, expected) if expected and top1 else None
    hit3 = (
        any(_path_matches(p, expected) for p in top_paths[:3])
        if expected and top_paths
        else None
    )
    bad_top = _path_matches(top1, reject) if reject and top1 else False

    return {
        "id": row.get("id"),
        "group": row.get("group"),
        "query": q,
        "latency_ms": latency_ms,
        "n_protocols": len(protos),
        "top1_path": top1,
        "top1_short": _basename(top1),
        "top1_confidence_pct": top_conf,
        "top_paths_short": [_basename(p) for p in top_paths],
        "ai_verdict": ai.get("ranking_verdict"),
        "ai_rating": ai.get("ranking_rating"),
        "ai_tags": ai.get("tags") or [],
        "top1_relevant": ai.get("top1_relevant"),
        "suggested_funnel_step": ai.get("suggested_funnel_step"),
        "engine_improvements": ai.get("engine_improvements_ru") or [],
        "ai_summary": ai.get("summary_ru"),
        "expected_hit1": hit1,
        "expected_hit3": hit3,
        "reject_in_top1": bad_top,
        "audience_inferred": data.get("audience_inferred"),
    }


def _summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in reports if not r.get("error")]
    err = [r for r in reports if r.get("error")]
    verdicts = Counter(str(r.get("ai_verdict") or "?") for r in ok)
    tags = Counter(t for r in ok for t in (r.get("ai_tags") or []))
    groups = Counter(str(r.get("group") or "?") for r in ok)
    improvements = Counter(
        imp[:140] for r in ok for imp in (r.get("engine_improvements") or []) if imp
    )
    hit1 = [r for r in ok if r.get("expected_hit1") is True]
    hit3 = [r for r in ok if r.get("expected_hit3") is True]
    labeled = [r for r in ok if r.get("expected_hit1") is not None]
    reject_bad = [r for r in ok if r.get("reject_in_top1")]
    low_rating = [r for r in ok if (r.get("ai_rating") or 5) <= 2]
    top1_bad = [r for r in ok if r.get("top1_relevant") is False]

    ratings = [int(r["ai_rating"]) for r in ok if r.get("ai_rating") is not None]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None

    return {
        "n_total": len(reports),
        "n_ok": len(ok),
        "n_error": len(err),
        "avg_ai_rating": avg_rating,
        "verdict_counts": dict(verdicts),
        "tag_counts": dict(tags.most_common(12)),
        "group_counts": dict(groups),
        "expected_hit1_pct": round(len(hit1) / len(labeled), 3) if labeled else None,
        "expected_hit3_pct": round(len(hit3) / len(labeled), 3) if labeled else None,
        "reject_in_top1_count": len(reject_bad),
        "top1_not_relevant_count": len(top1_bad),
        "low_rating_count": len(low_rating),
        "engine_improvements_top": improvements.most_common(8),
        "worst": sorted(
            ok,
            key=lambda r: (
                r.get("ai_rating") if r.get("ai_rating") is not None else 99,
                0 if r.get("top1_relevant") is False else 1,
                0 if r.get("reject_in_top1") else 1,
            ),
        )[:10],
        "errors": err,
    }


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

    print()
    print(md)
    print(f"JSONL: {args.out}")
    print(f"Markdown: {args.md}")
    return 0 if not summary.get("n_error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
