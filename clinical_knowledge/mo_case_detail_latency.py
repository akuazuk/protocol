"""Флаги быстрого пути разбора случая МО.

Критический путь GET /cases/{id}: warehouse findings + rubric без prior.
Live concordance/ICD и CSV-скан prior - только по запросу или когда findings пусты.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence


def _truthy(raw: str | None) -> bool | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _query_flag(params: Mapping[str, Any] | None, *names: str) -> bool | None:
    if not params:
        return None
    for name in names:
        if name not in params:
            continue
        parsed = _truthy(str(params.get(name)))
        if parsed is not None:
            return parsed
    return None


def findings_look_empty(findings: Sequence[Any] | None) -> bool:
    rows = findings if isinstance(findings, list) else []
    for item in rows:
        if not isinstance(item, dict):
            continue
        if item.get("passed"):
            continue
        code = str(item.get("code") or item.get("finding_code") or "").strip()
        title = str(item.get("title_ru") or item.get("title") or "").strip()
        if code or title:
            return False
    return True


def want_live_analyzers(
    *,
    query_params: Mapping[str, Any] | None = None,
    findings: Sequence[Any] | None = None,
) -> bool:
    """По умолчанию auto: live только если в витрине нет findings."""
    forced = _query_flag(query_params, "live_analyzers", "live")
    if forced is not None:
        return forced
    env = (os.environ.get("MO_CASE_DETAIL_LIVE_ANALYZERS") or "auto").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    if env in {"1", "true", "yes", "on"}:
        return True
    return findings_look_empty(findings)


def want_prior_clinical(*, query_params: Mapping[str, Any] | None = None) -> bool:
    """По умолчанию выкл: 90-дневный CSV-скан не блокирует drawer."""
    forced = _query_flag(query_params, "prior", "include_prior")
    if forced is not None:
        return forced
    env = (os.environ.get("MO_CASE_DETAIL_PRIOR") or "0").strip().lower()
    return env in {"1", "true", "yes", "on"}


def want_protocol_suggest_history(*, query_params: Mapping[str, Any] | None = None) -> bool:
    forced = _query_flag(query_params, "attach_history", "history")
    if forced is not None:
        return forced
    env = (os.environ.get("MO_PROTOCOL_SUGGEST_ATTACH_HISTORY") or "0").strip().lower()
    return env in {"1", "true", "yes", "on"}


def prewarm_protocol_suggest_match() -> dict[str, Any]:
    """Прогрев registry КП + пути матча (текст и ICD), чтобы cold open не ждал ~3 с."""
    from clinical_knowledge.loader import load_protocol_cards_registry
    from clinical_knowledge.protocol_match import (
        match_protocol_cards,
        match_protocol_cards_by_diagnosis_text,
    )

    cards = load_protocol_cards_registry()
    facts = {
        "patient_context": {"adult_or_child": "adult"},
        "consultation": {
            "icd10": ["G43.1"],
            "diagnosis_text": "Мигрень с аурой",
            "complaints": ["головная боль"],
            "conditions_hint": ["Мигрень с аурой"],
            "performed_exams": [],
        },
    }
    text_n = len(match_protocol_cards_by_diagnosis_text(facts, specialty_slug=None, limit=3) or [])
    icd_n = len(match_protocol_cards(facts, specialty_slug=None, limit=3, use_icd=True) or [])
    return {
        "ok": True,
        "cards": len(cards or []),
        "text_hits": text_n,
        "icd_hits": icd_n,
    }
