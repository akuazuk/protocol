"""Кэш и batch-навигация по Protocol Summary для UI поиска."""
from __future__ import annotations

import time
from typing import Any

_NAV_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_NAV_CACHE_TTL_SEC = 3600


def clear_protocol_nav_cache() -> None:
    _NAV_CACHE.clear()


def _cache_key(path: str, query: str, icd_codes: list[str] | None) -> str:
    icd_part = ",".join(sorted(c.strip().upper() for c in (icd_codes or []) if c))
    q = (query or "").strip()[:500]
    return f"{path.strip()}|{q}|{icd_part}"


def resolve_protocol_nav_cached(
    path: str,
    *,
    query: str = "",
    icd_codes: list[str] | None = None,
    allow_rich_fallback: bool = True,
) -> dict[str, Any]:
    """Nav payload с TTL-кэшем и метриками nav_ms / cache_hit."""
    from clinical_knowledge.search_funnel import resolve_protocol_nav

    t0 = time.perf_counter()
    key = _cache_key(path, query, icd_codes)
    now = time.time()
    cached = _NAV_CACHE.get(key)
    if cached and now - cached[0] < _NAV_CACHE_TTL_SEC:
        out = dict(cached[1])
        out["cache_hit"] = True
        out["nav_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    out = resolve_protocol_nav(
        path,
        query=query,
        icd_codes=icd_codes,
        allow_rich_fallback=allow_rich_fallback,
    )
    out = dict(out)
    out["cache_hit"] = False
    out["nav_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    _NAV_CACHE[key] = (now, dict(out))
    return out


def attach_protocol_nav_map(
    payload: dict[str, Any],
    *,
    query: str,
    icd_codes: list[str] | None = None,
    limit: int = 3,
) -> dict[str, Any]:
    """Вложить protocol_nav для top-N протоколов (только Summary, без rich fallback)."""
    protos: list[dict[str, Any]] = []
    llm = payload.get("llm_json")
    if isinstance(llm, dict):
        raw = llm.get("protocols") or []
        protos = [p for p in raw if isinstance(p, dict)]

    codes = list(icd_codes or [])
    if not codes:
        icd_payload = payload.get("icd") or {}
        if isinstance(icd_payload, dict):
            codes = list(icd_payload.get("codes_for_retrieval") or [])

    nav_map: dict[str, Any] = {}
    for pr in protos[: max(0, int(limit))]:
        pth = str(pr.get("path") or "").strip()
        if not pth or pth in nav_map:
            continue
        nav_map[pth] = resolve_protocol_nav_cached(
            pth,
            query=query,
            icd_codes=codes or None,
            allow_rich_fallback=False,
        )

    if nav_map:
        payload["protocol_nav"] = nav_map
    return payload
