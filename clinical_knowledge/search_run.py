"""POST /api/search/run - единая точка входа для воронки и tier S0/S1/S2."""
from __future__ import annotations

from typing import Any, Callable

from .search_tiering import apply_search_tier_flags, resolve_search_tier


def run_search_request(
    *,
    query: str,
    tier: str | None,
    step: int,
    context: dict[str, Any] | None,
    category_slugs: list[str] | None,
    icd_codes: list[str] | None,
    funnel_population: str | None,
    session_id: str | None,
    assist_fn: Callable[..., dict],
    funnel_fn: Callable[..., dict],
    protocols_by_icd_fn: Callable[..., dict],
) -> dict:
    """Маршрутизация: step>=0 → funnel; иначе tier → assist / protocols-by-icd."""
    q = (query or "").strip()
    if step >= 0:
        out = funnel_fn(
            query=q,
            step=int(step),
            context=dict(context or {}),
            category_slugs=list(category_slugs or []),
            session_id=session_id,
        )
        out.setdefault("search_tier", resolve_search_tier(tier))
        return out

    flags = apply_search_tier_flags(
        resolve_search_tier(tier),
        explicit_icd_codes=list(icd_codes or []),
        query=q,
    )
    if flags.get("error"):
        return {"ok": False, "error": flags["error"], "search_tier": flags.get("tier")}

    resolved = str(flags.get("tier") or "S1")

    if resolved == "S0" and icd_codes:
        payload = protocols_by_icd_fn(
            query=q,
            icd_codes=list(icd_codes),
            population=funnel_population,
            category_slugs=list(category_slugs or []),
            limit=8,
        )
        payload["search_tier"] = "S0"
        payload["ok"] = True
        return payload

    body = {
        "query": q,
        "category_slugs": list(category_slugs or []),
        "icd_codes": list(icd_codes or []),
        "funnel_population": funnel_population,
        "retrieve_only": bool(flags.get("retrieve_only")),
        "icd_fast_path": bool(flags.get("icd_fast_path")),
        "assist_full": bool(flags.get("assist_full")),
        "search_tier": resolved,
    }
    result = assist_fn(body)
    if isinstance(result, dict):
        result["search_tier"] = resolved
        result["ok"] = True
    return result
