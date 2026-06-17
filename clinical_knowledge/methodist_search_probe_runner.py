"""Batch probe поиска протоколов (режим методиста) — общая логика для CLI и API."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "search_methodist_probe.jsonl"

_POPULATION_LINE = {
    "adult": "Контекст подбора: взрослое население",
    "pediatric": "Контекст подбора: детское население",
    "pregnant": "Контекст подбора: беременные",
    "emergency": "Контекст подбора: неотложная помощь",
}


def load_probe_fixture(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or DEFAULT_FIXTURE
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def build_probe_query(row: dict[str, Any]) -> str:
    parts = [str(row.get("query") or "").strip()]
    pop = str(row.get("population") or "").strip().lower()
    if pop in _POPULATION_LINE:
        parts.append(_POPULATION_LINE[pop])
    icd = list(row.get("icd_codes") or [])
    if icd:
        parts.append("МКБ-10: " + ", ".join(str(c) for c in icd))
    return "\n".join(p for p in parts if p)


def _basename(path: str) -> str:
    p = (path or "").replace("\\", "/").strip()
    return p.rsplit("/", 1)[-1] if p else ""


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


def _infer_probe_icd_codes(query: str, row: dict[str, Any]) -> list[str]:
    """МКБ для воронки: из фикстуры или lexicon (как шаг 2 UI без Gemini)."""
    explicit = [str(c).strip() for c in (row.get("icd_codes") or []) if str(c).strip()]
    if explicit:
        return explicit
    from icd_mkb import analyze_query_for_icd
    from rag_server import clinical_query_for_rag

    analysis = analyze_query_for_icd(query, clinical_query_for_rag(query))
    codes: list[str] = []
    seen: set[str] = set()
    for bucket in ("detected", "suggested"):
        for r in (analysis.get(bucket) or []):
            if not isinstance(r, dict):
                continue
            c = str(r.get("code") or "").strip()
            if c and c not in seen:
                seen.add(c)
                codes.append(c)
    for c in (analysis.get("codes_for_retrieval") or []):
        cs = str(c).strip()
        if cs and cs not in seen:
            seen.add(cs)
            codes.append(cs)
    return codes[:8]


def run_single_probe(row: dict[str, Any]) -> dict[str, Any]:
    from clinical_knowledge.methodist_search_ai_review import build_deterministic_search_ai_review
    from clinical_knowledge.search_funnel import handle_search_funnel

    q = build_probe_query(row)
    slugs = [s for s in (row.get("category_slugs") or []) if isinstance(s, str)]
    icd_codes = _infer_probe_icd_codes(q, row)
    ctx: dict[str, Any] = {"population": row.get("population"), "icd_codes": icd_codes}
    if slugs:
        ctx["rubric_slugs"] = slugs
    t0 = __import__("time").perf_counter()
    try:
        import rag_server as rs

        rs._require_rag_loaded()
        body = handle_search_funnel(
            query=q,
            step=4,
            context=ctx,
            category_slugs=slugs or None,
            session_id="methodist-probe",
        )
        if body.get("error"):
            return {
                "id": row.get("id"),
                "group": row.get("group"),
                "query": q,
                "error": str(body.get("error")),
                "latency_ms": int((__import__("time").perf_counter() - t0) * 1000),
            }
        data = dict(body.get("assist") or {})
        if not data.get("icd") and body.get("icd"):
            data["icd"] = body["icd"]
    except Exception as exc:
        return {
            "id": row.get("id"),
            "group": row.get("group"),
            "query": q,
            "error": str(exc),
            "latency_ms": int((__import__("time").perf_counter() - t0) * 1000),
        }
    latency_ms = int((__import__("time").perf_counter() - t0) * 1000)
    top_paths = _top_paths(data)
    top1 = top_paths[0] if top_paths else ""
    protos = (data.get("llm_json") or {}).get("protocols") or []
    top_conf = None
    if protos and isinstance(protos[0], dict):
        try:
            top_conf = round(float(protos[0].get("confidence_score") or 0) * 100, 1)
        except (TypeError, ValueError):
            top_conf = None

    icd_codes = list(dict.fromkeys(icd_codes))
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
            "funnel_context": {"population": row.get("population"), "icd_codes": icd_codes},
            "audience_inferred": data.get("audience_inferred"),
            "hybrid_search": data.get("hybrid_search"),
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


def summarize_probe_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
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


def run_probe_batch(
    *,
    limit: int = 0,
    fixture_path: Path | None = None,
    group: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_probe_fixture(fixture_path)
    if group:
        rows = [r for r in rows if str(r.get("group") or "") == group]
    if limit > 0:
        rows = rows[:limit]
    reports = [run_single_probe(row) for row in rows]
    summary = summarize_probe_reports(reports)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["fixture"] = str(fixture_path or DEFAULT_FIXTURE)
    return reports, summary
