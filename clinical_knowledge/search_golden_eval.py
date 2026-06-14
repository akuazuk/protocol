"""Eval Hit@k / MRR для tests/fixtures/search_golden.jsonl (B3)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "tests" / "fixtures" / "search_golden.jsonl"


def load_search_golden(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or DEFAULT_GOLDEN
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def _basename_key(path: str) -> str:
    p = (path or "").replace("\\", "/").strip().lower()
    return p.rsplit("/", 1)[-1] if p else ""


def dedupe_protocol_paths(retrieved: list[dict[str, Any]], *, limit: int = 8) -> list[str]:
    """Уникальные PDF-пути в порядке score (как compact assist list)."""
    seen: set[str] = set()
    out: list[str] = []
    for row in sorted(retrieved, key=lambda r: -float(r.get("score") or 0)):
        p = str(row.get("path") or "").replace("\\", "/")
        if not p or p.startswith("summary://"):
            continue
        bk = _basename_key(p)
        if bk in seen:
            continue
        seen.add(bk)
        out.append(p)
        if len(out) >= limit:
            break
    return out


def _path_matches(path: str, fragments: list[str]) -> bool:
    pl = path.lower()
    for frag in fragments:
        f = str(frag or "").lower().strip()
        if f and f in pl:
            return True
    return False


def hit_at_k(protocol_paths: list[str], expected_fragments: list[str], k: int) -> bool:
    if not expected_fragments:
        return True
    top = protocol_paths[: max(1, k)]
    return any(_path_matches(p, expected_fragments) for p in top)


def reciprocal_rank(protocol_paths: list[str], expected_fragments: list[str]) -> float:
    if not expected_fragments:
        return 0.0
    for i, p in enumerate(protocol_paths, 1):
        if _path_matches(p, expected_fragments):
            return 1.0 / i
    return 0.0


def evaluate_search_golden_row(
    row: dict[str, Any],
    retrieve_fn: Callable[..., list[dict[str, Any]]],
    *,
    max_chunks: int = 12,
) -> dict[str, Any]:
    query = str(row.get("query") or "").strip()
    expect_empty = bool(row.get("expect_empty"))
    expected = list(row.get("expected_path_contains") or row.get("expected_path") or [])
    if isinstance(expected, str):
        expected = [expected]
    category_slugs = list(row.get("category_slugs") or [])
    icd_codes = list(row.get("icd_codes") or [])

    kwargs: dict[str, Any] = {
        "max_chunks": max_chunks,
        "max_per_path": 2,
        "routing_query": query,
    }
    if category_slugs:
        kwargs["user_category_slugs"] = category_slugs
        kwargs["category_boost"] = category_slugs
    if icd_codes:
        kwargs["icd_codes_for_lex"] = icd_codes

    retrieved = retrieve_fn(query, **kwargs)
    proto_paths = dedupe_protocol_paths(retrieved)

    if expect_empty:
        ok = len(retrieved) == 0
        return {
            "id": row.get("id"),
            "query": query,
            "ok": ok,
            "expect_empty": True,
            "hit1": ok,
            "hit3": ok,
            "mrr": 1.0 if ok else 0.0,
            "n_protocols": len(proto_paths),
            "top_paths": proto_paths[:3],
        }

    hit1 = hit_at_k(proto_paths, expected, 1)
    hit3 = hit_at_k(proto_paths, expected, 3)
    mrr = reciprocal_rank(proto_paths, expected)
    ok = hit3
    return {
        "id": row.get("id"),
        "query": query,
        "query_type": row.get("query_type"),
        "ok": ok,
        "hit1": hit1,
        "hit3": hit3,
        "mrr": round(mrr, 4),
        "n_protocols": len(proto_paths),
        "top_paths": proto_paths[:3],
        "expected": expected,
    }


def summarize_search_golden(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {"n": 0, "hit1": None, "hit3": None, "mrr": None}
    n = len(reports)
    hit1 = sum(1 for r in reports if r.get("hit1")) / n
    hit3 = sum(1 for r in reports if r.get("hit3")) / n
    mrr = sum(float(r.get("mrr") or 0) for r in reports) / n
    by_type: dict[str, list[dict[str, Any]]] = {}
    for r in reports:
        qt = str(r.get("query_type") or "unknown")
        by_type.setdefault(qt, []).append(r)
    type_stats = {}
    for qt, rows in sorted(by_type.items()):
        m = len(rows)
        type_stats[qt] = {
            "n": m,
            "hit1": round(sum(1 for x in rows if x.get("hit1")) / m, 3),
            "hit3": round(sum(1 for x in rows if x.get("hit3")) / m, 3),
        }
    return {
        "n": n,
        "hit1": round(hit1, 3),
        "hit3": round(hit3, 3),
        "mrr": round(mrr, 3),
        "pass_count": sum(1 for r in reports if r.get("ok")),
        "by_query_type": type_stats,
    }


def write_snapshot(summary: dict[str, Any], path: Path | None = None) -> Path:
    out = path or (ROOT / "data" / "ml" / "search_golden_snapshot.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out
