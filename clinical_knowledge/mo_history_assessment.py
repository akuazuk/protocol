"""Explicit history assessability contract for MO review.

This module does not change score weights. It separates warehouse availability,
existence of any prior visit, episode relevance and whether correction criteria
may be evaluated.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from clinical_knowledge.mo_history_continuity import evaluate_history_continuity

ENGINE = "mo_history_assessment_v1"
COMPARABLE_PLAN_FIELDS = frozenset(
    {
        "exam_recommendations",
        "treatment_recommendations",
        "recommendations",
        "plan",
    }
)
UNAVAILABLE_REASONS = frozenset(
    {
        "empty",
        "missing_patient_or_date",
        "no_warehouse",
        "schema_no_patient_key",
        "bad_case",
        "bad_db",
        "query_failed",
        "error",
    }
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _has_comparable_plan(value: Mapping[str, Any] | None) -> bool:
    if not isinstance(value, Mapping):
        return False
    clinical = value.get("clinical") if isinstance(value.get("clinical"), Mapping) else value
    return any(str(clinical.get(field) or "").strip() for field in COMPARABLE_PLAN_FIELDS)


def build_history_assessment_context(
    *,
    history_bundle: Mapping[str, Any] | None,
    current_code: str = "",
    current_text: str = "",
    cutoff_at: str = "",
    document_prior: Mapping[str, Any] | None = None,
    episode_deep: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the public, text-free history assessment contract."""

    bundle = _mapping(history_bundle)
    summary = _mapping(bundle.get("summary"))
    reason = str(bundle.get("reason") or "")
    status = str(bundle.get("status") or "")
    query_ok = bundle.get("ok")
    history_available = bool(bundle) and query_ok is not False and reason not in UNAVAILABLE_REASONS
    if status in {"error", "unavailable"}:
        history_available = False
    any_prior_exists = int(summary.get("n_visits") or 0) > 0 or bool(document_prior)

    deep = _mapping(episode_deep)
    continuity = _mapping(deep.get("continuity"))
    if not continuity and bundle:
        continuity = evaluate_history_continuity(
            current_code=current_code,
            current_text=current_text,
            history_bundle=bundle,
        )
    relevant_episode_prior_exists = bool(
        continuity.get("known_episode")
        and (
            int(deep.get("prior_n_loaded") or 0) > 0
            or bool(deep.get("prior_clinical"))
            or bool(continuity.get("last_matched_date"))
        )
    )

    episode_prior = (
        deep.get("prior_clinical")
        if isinstance(deep.get("prior_clinical"), Mapping)
        else None
    )
    explicitly_relevant_document = bool(
        isinstance(document_prior, Mapping)
        and document_prior.get("episode_relevant") is True
    )
    comparable_prior = _has_comparable_plan(episode_prior) or (
        explicitly_relevant_document and _has_comparable_plan(document_prior)
    )
    correction_assessable = bool(relevant_episode_prior_exists and comparable_prior)

    exclusions: list[str] = []
    if not history_available:
        exclusions.append("history_unavailable")
    if not any_prior_exists:
        exclusions.append("no_prior_visit")
    elif not relevant_episode_prior_exists:
        exclusions.append("prior_not_same_episode")
    elif not comparable_prior:
        exclusions.append("prior_has_no_comparable_plan")

    if not status:
        if not history_available:
            status = "unavailable" if reason in UNAVAILABLE_REASONS else "empty"
        elif any_prior_exists:
            status = "has_priors"
        else:
            status = "empty"
    return {
        "contract_version": 1,
        "engine": ENGINE,
        "ok": history_available,
        "status": status,
        "cutoff_at": str(cutoff_at or bundle.get("cutoff_at") or "") or None,
        "history_available": history_available,
        "any_prior_exists": any_prior_exists,
        "relevant_episode_prior_exists": relevant_episode_prior_exists,
        "correction_assessable": correction_assessable,
        "exclusion_reason_codes": exclusions,
        "sources": {
            "longitudinal": {
                "available": history_available,
                "prior_n": int(summary.get("n_visits") or 0),
            },
            "document": {
                "available": bool(document_prior),
                "episode_relevant": explicitly_relevant_document,
            },
        },
        "primary": False,
        "shadow": True,
    }

