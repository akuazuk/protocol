"""Быстрый L0-скрининг КЗ для МИС (без полного RAG/LLM-конвейера)."""
from __future__ import annotations

from typing import Any

from .cisz_readiness import attach_cisz_readiness
from .consult_analysis import analyze_consultation_text
from .compliance_gate import evaluate_send_gate_from_compliance
from .fhir_bundle_adapter import bundle_to_consultation_text


def run_compliance_screen(
    *,
    text: str | None = None,
    bundle: dict[str, Any] | None = None,
    consultation_id: str = "screen",
) -> dict[str, Any]:
    """Структурный разбор + send_gate; без загрузки PDF и без LLM-критериев."""
    if bundle and not text:
        text = bundle_to_consultation_text(bundle)
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Пустой текст заключения")

    sa = analyze_consultation_text(
        raw,
        consultation_id=consultation_id,
        with_markdown=False,
        match_limit=3,
        analysis_mode="legacy",
    )
    comp = sa.get("compliance") or {}
    send_gate = evaluate_send_gate_from_compliance(comp)
    comp["send_gate"] = send_gate
    out = {
        "ok": True,
        "screen_level": "L0",
        "review_tier": "L0",
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
        "critical_issues_count": len(comp.get("critical_issues") or []),
        "matched_protocols_count": len(sa.get("matches") or []),
        "llm_used": False,
        "rag_used": False,
        "source": "fhir_bundle" if bundle else "text",
    }
    return attach_cisz_readiness(out, bundle=bundle, text=raw if not bundle else None)
