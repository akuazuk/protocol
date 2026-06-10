"""Уровни проверки КЗ L0/L1/L2 для массового потока МИС."""
from __future__ import annotations

import os
from typing import Any

from .cisz_readiness import attach_cisz_readiness
from .consult_analysis import analyze_consultation_text
from .consult_screen import run_compliance_screen
from .compliance_gate import evaluate_send_gate_from_compliance
from .fhir_bundle_adapter import bundle_to_consultation_text


VALID_TIERS = frozenset({"L0", "L1", "L2"})


def resolve_tier(tier: str | None = None) -> str:
    """L0 - скрининг; L1 - полный structured без RAG/LLM; L2 - полный pipeline."""
    raw = (tier or os.environ.get("CONSULT_REVIEW_TIER", "L2")).strip().upper()
    if raw not in VALID_TIERS:
        return "L2"
    return raw


def run_l1_structured_review(
    *,
    text: str,
    consultation_id: str = "l1",
    demographics_meta: dict[str, Any] | None = None,
    specialty_slug: str | None = None,
    analysis_mode: str | None = None,
) -> dict[str, Any]:
    """L1: полный детерминированный разбор + compliance, без RAG и без LLM-критериев."""
    mode = analysis_mode or (
        os.environ.get("PROTOCOL_SUMMARY_MODE")
        if os.environ.get("PROTOCOL_SUMMARY_ENABLED", "").strip().lower()
        in ("1", "true", "yes")
        else "legacy"
    )
    sa = analyze_consultation_text(
        text,
        consultation_id=consultation_id,
        demographics_meta=demographics_meta,
        specialty_slug=specialty_slug,
        with_markdown=True,
        analysis_mode=mode,
    )
    comp = sa.get("compliance") or {}
    send_gate = evaluate_send_gate_from_compliance(comp)
    comp["send_gate"] = send_gate
    return {
        "ok": True,
        "review_tier": "L1",
        "consultation_id": comp.get("consultation_id"),
        "overall_score": comp.get("overall_score"),
        "overall_status": comp.get("overall_status"),
        "confidence_score": comp.get("confidence_score"),
        "send_gate": send_gate,
        "structured_analysis": {
            "document": sa.get("document"),
            "matches": sa.get("matches"),
            "compliance": comp,
            "rubric_specifics": sa.get("rubric_specifics"),
        },
        "report_markdown": sa.get("report_markdown"),
        "report_html": sa.get("report_html"),
        "matched_protocols_count": len(sa.get("matches") or []),
        "critical_issues_count": len(comp.get("critical_issues") or []),
        "llm_used": False,
        "rag_used": False,
    }


def run_consult_by_tier(
    *,
    tier: str | None,
    text: str | None = None,
    bundle: dict[str, Any] | None = None,
    consultation_id: str = "consult",
    category_slugs: str = "",
    demographics_meta: dict[str, Any] | None = None,
    specialty_slug: str | None = None,
) -> dict[str, Any]:
    """Маршрутизация по уровню. L2 возвращает маркер для полного pipeline."""
    level = resolve_tier(tier)
    if bundle and not text:
        text = bundle_to_consultation_text(bundle)
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Пустой текст заключения")

    if level == "L0":
        out = run_compliance_screen(
            text=raw,
            bundle=None,
            consultation_id=consultation_id,
        )
        out["review_tier"] = "L0"
        return attach_cisz_readiness(out, bundle=bundle, text=raw if not bundle else None)

    if level == "L1":
        out = run_l1_structured_review(
            text=raw,
            consultation_id=consultation_id,
            demographics_meta=demographics_meta,
            specialty_slug=specialty_slug,
        )
        return attach_cisz_readiness(out, bundle=bundle, text=raw if not bundle else None)

    return {
        "ok": True,
        "review_tier": "L2",
        "delegate_full_pipeline": True,
        "category_slugs": category_slugs,
        "text": raw,
        "bundle": bundle,
        "consultation_id": consultation_id,
    }
