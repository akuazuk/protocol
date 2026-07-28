"""LLM multi-pass extraction for Protocol Summary."""
from __future__ import annotations

import os
from typing import Any, Callable

from .llm_json import parse_json_loose
from .llm_merger import merge_to_protocol_summary
from .llm_prompts import (
    SYSTEM_CONDITION_BLOCK,
    SYSTEM_SKELETON,
    prompt_condition_block,
    prompt_skeleton,
)
from .quote_validator import validate_quotes_in_payload
from .schema import ProtocolSummary
from .source_text import section_text_blob
from .structured_fallback import build_structured_summary


def _strict_enabled() -> bool:
    return os.environ.get("PROTOCOL_LLM_STRICT", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _gemini_available() -> bool:
    try:
        from clinical_knowledge.gemini_lite import gemini_available

        return gemini_available()
    except Exception:
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def _call_llm(
    system: str,
    user: str,
    *,
    generate_fn: Callable | None = None,
    model: Any = None,
) -> dict[str, Any] | list[Any] | None:
    prompt = f"{system}\n\n{user}"
    try:
        if generate_fn is not None and model is not None:
            resp = generate_fn(model, prompt)
            text = resp if isinstance(resp, str) else str(resp)
        else:
            from clinical_knowledge.gemini_lite import (
                generate_lite_json,
                gemini_available,
                get_lite_gemini_model,
            )

            if not gemini_available():
                if _strict_enabled():
                    raise RuntimeError("Нейросетевая модель недоступна в strict-режиме")
                return None
            model = model or get_lite_gemini_model()
            text = generate_lite_json(model, prompt)
    except Exception:
        if _strict_enabled():
            raise
        return None
    data = parse_json_loose(text or "")
    return data if isinstance(data, (dict, list)) else None


def extract_protocol_summary_llm(
    doc: dict[str, Any],
    *,
    generate_fn: Callable | None = None,
    model: Any = None,
    use_llm: bool | None = None,
) -> ProtocolSummary:
    """Extract summary: LLM multi-pass if key present, else structured fallback."""
    if use_llm is None:
        use_llm = _gemini_available()
    if not use_llm:
        return build_structured_summary(doc)

    protocol_id = str(doc.get("protocol_id") or "")
    catalog_icd = [str(c) for c in (doc.get("icd10_primary") or [])]
    skeleton_data = _call_llm(
        SYSTEM_SKELETON,
        prompt_skeleton(doc, catalog_icd),
        generate_fn=generate_fn,
        model=model,
    )
    if not isinstance(skeleton_data, dict) or not skeleton_data.get("conditions"):
        if _strict_enabled():
            raise RuntimeError(
                f"Не получена структура протокола {protocol_id or '<unknown>'}",
            )
        fb = build_structured_summary(doc)
        fb.extraction_metadata.notes.append("llm_skeleton_failed_fallback")
        return fb

    condition_blocks: dict[str, list[dict[str, Any]]] = {}
    source_all = section_text_blob(doc, ["classification", "diagnostics", "treatment", "routing", "other"])

    for sk in skeleton_data.get("conditions") or []:
        if not isinstance(sk, dict):
            continue
        cid = str(sk.get("condition_id") or "")
        blocks: list[dict[str, Any]] = []
        for block_name in ("diagnostics", "treatment", "routing"):
            raw = _call_llm(
                SYSTEM_CONDITION_BLOCK,
                prompt_condition_block(doc, sk, block_name),
                generate_fn=generate_fn,
                model=model,
            )
            if isinstance(raw, dict):
                issues = validate_quotes_in_payload(raw, source_all)
                if issues:
                    raw["_quote_issues"] = issues
                blocks.append(raw)
        condition_blocks[cid] = blocks

    summary = merge_to_protocol_summary(
        doc,
        skeleton_data,
        condition_blocks,
        extractor="llm_extractor",
        extractor_version="1.1-regimen",
        extraction_status="llm_extracted",
    )
    summary.extraction_metadata.notes.append(f"conditions={len(summary.conditions)}")
    return summary
