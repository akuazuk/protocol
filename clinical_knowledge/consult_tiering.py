"""L1 structured review with deterministic alignment cards."""
from __future__ import annotations

import os
from typing import Any

from .cisz_readiness import attach_cisz_readiness
from .consult_analysis import analyze_consultation_text
from .consult_parser import parse_consultation
from .consult_screen import run_compliance_screen
from .compliance_gate import evaluate_send_gate_from_compliance
from .fhir_bundle_adapter import bundle_to_consultation_text


VALID_TIERS = frozenset({"L0", "L1", "L2"})


def resolve_tier(tier: str | None = None) -> str:
    """L0 - скрининг; L1 - полный structured без RAG/LLM; L2 - полный pipeline."""
    raw = (tier or os.environ.get("CONSULT_REVIEW_TIER", "L1")).strip().upper()
    if raw not in VALID_TIERS:
        return "L2"
    return raw


def _l1_get_chunks(path: str) -> list[dict[str, Any]]:
    try:
        import rag_server as rs

        return rs.get_rich_chunks_for_consult(path) or []
    except Exception:
        return []


def run_l1_structured_review(
    *,
    text: str,
    consultation_id: str = "l1",
    demographics_meta: dict[str, Any] | None = None,
    specialty_slug: str | None = None,
    analysis_mode: str | None = None,
    skip_alignment: bool = False,
    max_alignment_paths: int | None = None,
    get_chunks: Any | None = None,
) -> dict[str, Any]:
    """L1: structured + alignment, без RAG и без LLM-критериев."""
    mode = analysis_mode or (
        os.environ.get("PROTOCOL_SUMMARY_MODE")
        if os.environ.get("PROTOCOL_SUMMARY_ENABLED", "").strip().lower()
        in ("1", "true", "yes")
        else "legacy"
    )
    doc = parse_consultation(
        text,
        consultation_id=consultation_id,
        demographics_meta=demographics_meta,
    )
    sa = analyze_consultation_text(
        text,
        consultation_id=consultation_id,
        demographics_meta=demographics_meta,
        specialty_slug=specialty_slug,
        with_markdown=False,
        analysis_mode=mode,
        doc=doc,
    )
    comp = sa.get("compliance") or {}
    send_gate = evaluate_send_gate_from_compliance(comp)
    comp["send_gate"] = send_gate

    structured_analysis = {
        "document": sa.get("document"),
        "matches": sa.get("matches"),
        "compliance": comp,
        "rubric_specifics": sa.get("rubric_specifics"),
    }

    review: dict[str, Any] = {
        "summary_ru": "",
        "criteria": [],
        "limitations_ru": "",
        "disclaimer_ru": "Оценка ориентировочная; не замена МЭЭ и очной экспертизы.",
    }
    alignment_result = None
    chunk_fn = get_chunks if get_chunks is not None else _l1_get_chunks

    if (
        not skip_alignment
        and os.environ.get("CONSULT_ALIGNMENT_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
    ):
        try:
            from .consult_alignment import (
                append_alignment_evidence,
                build_consult_alignment,
                merge_alignment_into_review,
                sync_structured_with_alignment,
            )
            from .consult_retrieval import unify_consult_protocol_paths

            matches = sa.get("matches") or []
            match_paths = [
                str(m.get("source_path") or "")
                for m in matches
                if isinstance(m, dict) and m.get("source_path")
            ]
            icd_codes = [
                str(d.icd10_code or "")
                for d in (doc.diagnoses or [])
                if getattr(d, "icd10_code", None)
            ]
            alignment_paths = unify_consult_protocol_paths(
                target_paths=match_paths,
                rules_paths=match_paths,
                rag_paths=[],
            )
            if max_alignment_paths is not None and max_alignment_paths > 0:
                alignment_paths = alignment_paths[:max_alignment_paths]
            protocol_matches = [
                {
                    "title": m.get("title"),
                    "source_path": m.get("source_path"),
                    "match_score": m.get("match_score"),
                }
                for m in matches
                if isinstance(m, dict)
            ]
            alignment_result = build_consult_alignment(
                doc,
                protocol_paths=alignment_paths,
                icd_codes=icd_codes,
                get_chunks=chunk_fn,
                query=" ".join(icd_codes[:4]),
                protocol_matches=protocol_matches,
                specialty_label=doc.doctor_specialty,
            )
            merge_alignment_into_review(review, alignment_result)
            sync_structured_with_alignment(structured_analysis, alignment_result)
            append_alignment_evidence(structured_analysis, alignment_result)
            if alignment_result.get("limitations_ru"):
                review["limitations_ru"] = alignment_result["limitations_ru"]
        except Exception:
            alignment_result = None

    return {
        "ok": True,
        "review_tier": "L1",
        "review": review,
        "consultation_id": comp.get("consultation_id"),
        "overall_score": comp.get("overall_score"),
        "overall_status": comp.get("overall_status"),
        "confidence_score": comp.get("confidence_score"),
        "send_gate": send_gate,
        "structured_analysis": structured_analysis,
        "alignment": alignment_result,
        "report_markdown": sa.get("report_markdown"),
        "report_html": sa.get("report_html"),
        "matched_protocols_count": len(sa.get("matches") or []),
        "critical_issues_count": len(comp.get("critical_issues") or []),
        "llm_used": False,
        "rag_used": False,
        "criteria_source": review.get("criteria_source"),
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

    from .patient_upload_classifier import check_consult_document, build_consult_upload_mismatch_response

    mismatch = check_consult_document(raw, consultation_id=consultation_id)
    if mismatch:
        return build_consult_upload_mismatch_response(
            mismatch,
            consultation_id=consultation_id,
            review_tier=level,
        )

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
