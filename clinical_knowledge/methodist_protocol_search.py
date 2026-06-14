"""Autocomplete протоколов для кабинета методиста (поиск + разметка retrieval_fix)."""
from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
INDEX_CSV = ROOT / "index.csv"


def _score_match(query: str, *candidates: str) -> float:
    q = query.lower().strip()
    if not q:
        return 0.0
    best = 0.0
    for raw in candidates:
        t = (raw or "").lower()
        if not t:
            continue
        if q in t:
            best = max(best, 0.85 + min(0.14, len(q) / max(len(t), 1)))
        else:
            best = max(best, SequenceMatcher(None, q, t[:200]).ratio() * 0.7)
    return best


def search_catalog_protocols(query: str, *, limit: int = 10) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if len(q) < 2:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()

    if INDEX_CSV.is_file():
        with INDEX_CSV.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rel = (row.get("relative_path") or "").strip()
                if not rel or rel in seen:
                    continue
                fn = (row.get("filename") or Path(rel).name).strip()
                cat = (row.get("category") or "").strip()
                sc = _score_match(q, rel, fn, cat)
                if sc < 0.35:
                    continue
                seen.add(rel)
                path = rel if rel.startswith("minzdrav_protocols/") else f"minzdrav_protocols/{rel.lstrip('/')}"
                scored.append(
                    (
                        sc,
                        {
                            "path": path,
                            "title": fn,
                            "category": cat,
                            "source": "index_csv",
                        },
                    )
                )

    try:
        from clinical_knowledge.protocol_summary.loader import load_protocol_summaries

        for summary in load_protocol_summaries(usable_only=False):
            lp = (summary.source.local_path or "").strip()
            if not lp:
                continue
            path = lp if lp.startswith("minzdrav_protocols/") else f"minzdrav_protocols/{lp.lstrip('/')}"
            if path in seen:
                continue
            title = summary.source.title or summary.protocol_id
            sc = _score_match(q, title, path, summary.protocol_id)
            if sc < 0.35:
                continue
            seen.add(path)
            scored.append(
                (
                    sc,
                    {
                        "path": path,
                        "title": title,
                        "protocol_id": summary.protocol_id,
                        "source": "summary",
                    },
                )
            )
    except Exception:
        pass

    scored.sort(key=lambda x: (-x[0], x[1].get("title") or ""))
    return [item for _, item in scored[: max(1, min(limit, 20))]]
