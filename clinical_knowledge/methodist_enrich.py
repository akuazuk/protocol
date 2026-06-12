"""Обогащение ответов L0/L1 для кабинета методиста (правила + метаданные уровня)."""
from __future__ import annotations

import os
from typing import Any

TIER_META: dict[str, dict[str, Any]] = {
    "L0": {
        "label_ru": "L0 — быстрый скрининг",
        "latency_hint_ru": "≈5–15 с",
        "checks_ru": [
            "8 блоков структурного compliance (оформление, диагноз, лечение…)",
            "Краткий подбор протоколов (до 3 карточек)",
            "Детерминированные правила протокола",
        ],
        "not_included_ru": [
            "RAG по PDF протоколов",
            "LLM-критерии и цитаты модели",
            "Полный structured-отчёт с таблицами",
        ],
    },
    "L1": {
        "label_ru": "L1 — структурный разбор",
        "latency_hint_ru": "≈15–45 с",
        "checks_ru": [
            "Полный детерминированный разбор КЗ (8 блоков)",
            "Критические замечания по протоколам",
            "Подбор протоколов по рубрике/диагнозу",
            "Детерминированные правила протокола",
        ],
        "not_included_ru": [
            "RAG по PDF протоколов",
            "LLM-критерии и пояснения модели",
        ],
    },
    "L2": {
        "label_ru": "L2 — полный RAG + LLM",
        "latency_hint_ru": "≈2–5 мин",
        "checks_ru": [
            "Всё из L1",
            "RAG: отбор фрагментов PDF протоколов",
            "LLM-критерии с цитатами из КЗ и протокола",
            "Гибридный итог (структура + правила + LLM)",
        ],
        "not_included_ru": [],
    },
}


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _resolve_specialty(category_slugs: str, result: dict[str, Any]) -> str | None:
    slugs = [s.strip() for s in (category_slugs or "").split(",") if s.strip()]
    if len(slugs) == 1:
        return slugs[0]
    sa = result.get("structured_analysis") or {}
    for m in sa.get("matches") or result.get("matches") or []:
        if isinstance(m, dict):
            slug = (m.get("specialty_slug") or m.get("rubric_slug") or "").strip()
            if slug:
                return slug
    env_slug = (os.environ.get("CONSULT_RULE_CHECK_SPECIALTY") or "").strip()
    return env_slug or None


def attach_clinical_rules(
    result: dict[str, Any],
    *,
    full_text: str,
    category_slugs: str = "",
    match_limit: int = 8,
    demographics_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Добавляет clinical_rules, если их ещё нет (L0/L1 без полного pipeline)."""
    if result.get("clinical_rules"):
        return result
    if not _env_bool("CONSULT_RULE_CHECK", True):
        return result
    try:
        from clinical_knowledge import (
            extract_consult_facts_heuristic,
            match_protocol_cards,
            run_rule_checker,
        )
    except ImportError:
        return result

    text = (full_text or "").strip()
    if not text:
        return result

    sa = result.get("structured_analysis") or {}
    doc = sa.get("document") or {}
    demo = demographics_meta or doc.get("demographics_meta") or {}

    facts = extract_consult_facts_heuristic(text, demographics_meta=demo)
    cons = facts.setdefault("consultation", {})
    icd_from_doc = doc.get("icd10") or doc.get("icd_codes") or []
    cons["icd10"] = list(
        dict.fromkeys((cons.get("icd10") or []) + [str(c).upper() for c in icd_from_doc if c])
    )

    specialty = _resolve_specialty(category_slugs, result)
    try:
        matched = match_protocol_cards(facts, specialty_slug=specialty, limit=match_limit)
        rules = run_rule_checker(facts, matched_protocols=matched)
    except Exception as exc:
        result = dict(result)
        result["clinical_rules"] = {
            "consult_facts": facts,
            "matched_protocols": [],
            "rules_check": {"error": str(exc)[:240], "rules": [], "findings": []},
            "specialty_scope": specialty or "all_catalog",
        }
        return result

    result = dict(result)
    result["clinical_rules"] = {
        "consult_facts": facts,
        "matched_protocols": matched,
        "rules_check": rules,
        "specialty_scope": specialty or "all_catalog",
    }
    return result


def enrich_methodist_tier_payload(
    result: dict[str, Any],
    *,
    tier: str,
    full_text: str,
    category_slugs: str = "",
    latency_ms: int | None = None,
) -> dict[str, Any]:
    """Метаданные уровня + clinical_rules для L0/L1."""
    level = (tier or result.get("review_tier") or "L2").strip().upper()
    out = dict(result)
    out["review_tier"] = level

    meta = dict(TIER_META.get(level, TIER_META["L2"]))
    if latency_ms is not None:
        meta["latency_ms"] = latency_ms
        if latency_ms >= 1000:
            meta["latency_label_ru"] = f"{latency_ms / 1000:.1f} с"
        else:
            meta["latency_label_ru"] = f"{latency_ms} мс"
    out["methodist_tier_meta"] = meta

    if level in ("L0", "L1"):
        rule_limit = 4 if level == "L0" else 8
        out = attach_clinical_rules(
            out,
            full_text=full_text,
            category_slugs=category_slugs,
            match_limit=rule_limit,
        )
    return out
