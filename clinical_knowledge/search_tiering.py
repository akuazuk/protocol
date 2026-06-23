"""Уровни поиска протоколов S0/S1/S2 (аналог L0/L1/L2 для КЗ)."""
from __future__ import annotations

import os
import re
from typing import Any

VALID_SEARCH_TIERS = frozenset({"S0", "S1", "S2"})


def env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def resolve_search_tier(tier: str | None = None) -> str:
    """S0 - только индекс МКБ; S1 - retrieve без LLM; S2 - полный assist."""
    raw = (tier or os.environ.get("PROTOCOL_SEARCH_TIER", "S1")).strip().upper()
    if raw not in VALID_SEARCH_TIERS:
        return "S1"
    return raw


def query_has_icd_code(text: str) -> bool:
    return bool(
        re.search(r"\b[A-TV-ZА-ЯЁ]\s*\d{2}(?:\s*[.,/\-]\s*\d{1,4})?\b", text or "", re.I)
    )


def apply_search_tier_flags(
    tier: str,
    *,
    explicit_icd_codes: list[str] | None = None,
    query: str = "",
) -> dict[str, Any]:
    """Флаги для AssistIn из уровня S0/S1/S2."""
    t = resolve_search_tier(tier)
    icd_codes = list(explicit_icd_codes or [])
    has_icd = bool(icd_codes) or query_has_icd_code(query)
    if t == "S0":
        if not has_icd:
            return {
                "tier": "S0",
                "error": "S0: укажите код МКБ-10 в запросе или выберите код на шаге воронки.",
            }
        return {
            "tier": "S0",
            "retrieve_only": True,
            "icd_fast_path": True,
            "assist_full": False,
            "require_icd_fast": True,
        }
    if t == "S2":
        return {
            "tier": "S2",
            "retrieve_only": False,
            "icd_fast_path": bool(icd_codes),
            "assist_full": False,
            "require_icd_fast": False,
        }
    return {
        "tier": "S1",
        "retrieve_only": True,
        "icd_fast_path": bool(icd_codes) or has_icd,
        "assist_full": False,
        "require_icd_fast": False,
    }


def search_require_allowlist() -> bool:
    if env_bool("RAG_SEARCH_REQUIRE_ALLOWLIST", False):
        return True
    return env_bool("RENDER", False) and env_bool("RAG_SEARCH_REQUIRE_ALLOWLIST_ON_RENDER", True)


def build_search_path_allowlist(
    *,
    path_allowlist: list[str] | None,
    icd_lookup_allowlist: list[str] | None,
    icd_codes: list[str] | None,
    path_boost: list[str] | None,
    search_ctx: dict[str, Any] | None,
) -> list[str] | None:
    """Собрать allowlist для retrieve (не сканировать весь корпус на Render)."""
    merged: list[str] = []
    for src in (
        path_allowlist or [],
        icd_lookup_allowlist or [],
        path_boost or [],
        (search_ctx or {}).get("path_allowlist") or [],
        (search_ctx or {}).get("path_boost") or [],
    ):
        for p in src:
            ps = str(p or "").replace("\\", "/").strip()
            if ps and ps not in merged:
                merged.append(ps)
    if not merged and icd_codes:
        try:
            from clinical_knowledge.protocol_summary.icd_index import find_catalog_paths_by_icd_codes

            for p in find_catalog_paths_by_icd_codes(icd_codes, limit=15):
                if p not in merged:
                    merged.append(p)
        except Exception:
            pass
    return merged[:15] if merged else None
