"""Autocomplete протоколов для кабинета методиста (поиск + разметка retrieval_fix)."""
from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from clinical_knowledge.protocol_links import protocol_display_name, protocol_nav_api_path

ROOT = Path(__file__).resolve().parent.parent
INDEX_CSV = ROOT / "index.csv"
_ICD_RE = re.compile(r"\b([A-ZА-Я]\d{2}(?:\.\d{1,2})?)\b", re.I)


def _score_match(query: str, *candidates: str) -> float:
    q = query.lower().strip()
    if not q:
        return 0.0
    best = 0.0
    tokens = [t for t in re.findall(r"[а-яa-z0-9.]{3,}", q, flags=re.I) if t]
    for raw in candidates:
        t = (raw or "").lower()
        if not t:
            continue
        if q in t:
            best = max(best, 0.88 + min(0.11, len(q) / max(len(t), 1)))
            continue
        token_hits = sum(1 for tok in tokens if tok in t)
        if tokens and token_hits:
            best = max(best, 0.45 + 0.4 * (token_hits / len(tokens)))
        best = max(best, SequenceMatcher(None, q, t[:240]).ratio() * 0.75)
        # Бонус за МКБ в пути/названии.
        for m in _ICD_RE.finditer(query):
            code = m.group(1).upper().replace("А", "A").replace("В", "B").replace("С", "C")
            if code.lower() in t or code[:3].lower() in t:
                best = max(best, 0.92)
    return best


def search_catalog_protocols(query: str, *, limit: int = 10) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if len(q) < 2:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    limit_n = max(1, min(int(limit or 10), 30))

    if INDEX_CSV.is_file():
        with INDEX_CSV.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rel = (row.get("relative_path") or "").strip()
                if not rel or rel in seen:
                    continue
                fn = (row.get("filename") or Path(rel).name).strip()
                cat = (row.get("category") or "").strip()
                path = rel if rel.startswith("minzdrav_protocols/") else f"minzdrav_protocols/{rel.lstrip('/')}"
                title = protocol_display_name(path, fallback=fn, prefer_filename_if_truncated=True)
                try:
                    from clinical_knowledge.protocol_content_index import content_text_for_path

                    body = content_text_for_path(path)
                except Exception:
                    body = ""
                sc = _score_match(q, title, rel, fn, cat, body)
                if sc < 0.32:
                    continue
                seen.add(rel)
                seen.add(path)
                scored.append(
                    (
                        sc,
                        {
                            "path": path,
                            "title": title,
                            "category": cat,
                            "source": "index_csv",
                            "viewer_url": protocol_nav_api_path(path),
                            "score": round(sc, 3),
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
            raw_title = summary.source.title or summary.protocol_id
            title = protocol_display_name(
                path,
                fallback=str(raw_title or ""),
                registry_title=str(raw_title or ""),
                prefer_filename_if_truncated=True,
            )
            try:
                from clinical_knowledge.protocol_content_index import content_text_for_path

                body = content_text_for_path(path)
            except Exception:
                body = ""
            sc = _score_match(q, title, path, summary.protocol_id, str(raw_title or ""), body)
            if sc < 0.32:
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
                        "viewer_url": protocol_nav_api_path(path),
                        "score": round(sc, 3),
                    },
                )
            )
    except Exception:
        pass

    scored.sort(key=lambda x: (-x[0], x[1].get("title") or ""))
    return [item for _, item in scored[:limit_n]]
