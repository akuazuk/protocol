"""RAG-подбор протоколов для B2C: corpus chunks + vector embeddings (как в КЗ L2 и поиске)."""
from __future__ import annotations

import os
import re
from typing import Any

from .patient_flags import patient_rag_retrieval_enabled


def _patient_rag_max_chunks() -> int:
    try:
        return max(4, min(20, int(os.environ.get("PATIENT_RAG_MAX_CHUNKS", "10"))))
    except (TypeError, ValueError):
        return 10


def _patient_rag_max_paths() -> int:
    try:
        return max(1, min(6, int(os.environ.get("PATIENT_RAG_MAX_PATHS", "4"))))
    except (TypeError, ValueError):
        return 4


def _diag_block(text: str) -> str:
    m = re.search(
        r"диагноз[^:\n]*[:\-]\s*(.+?)(?:\n\s*\n|рекомендац|обследован|лечени|назначен|$)",
        text,
        re.I | re.S,
    )
    return (m.group(1) if m else "") or text


def retrieve_patient_protocol_context(
    *,
    kz_text: str,
    demographics_meta: dict[str, Any] | None = None,
    specialty_slug: str | None = None,
) -> dict[str, Any]:
    """
    Подбор путей протоколов и top-чанков через retrieve() с embed-rerank.
    Не бросает исключений - при сбое возвращает пустой контекст.
    """
    empty: dict[str, Any] = {
        "paths": [],
        "retrieved": [],
        "icd_codes": [],
        "rag_used": False,
        "vector_used": False,
    }
    if not patient_rag_retrieval_enabled():
        return empty
    raw = (kz_text or "").strip()
    if not raw:
        return empty

    try:
        import rag_server as rs
        from clinical_knowledge.consult_parser import _detect_specialty
        from clinical_knowledge.consult_retrieval import (
            consult_target_protocol_paths,
            filter_retrieval_rows_by_paths,
        )
        from clinical_knowledge.diagnosis_icd import lookup_disease_icd, prioritize_codes
        from clinical_knowledge.rubric_extractors import specialty_to_rubric
    except Exception:
        return empty

    try:
        rs._require_rag_loaded(max_wait_sec=float(os.environ.get("PATIENT_RAG_LOAD_WAIT_SEC", "20")))
    except Exception:
        return empty

    diag_text = _diag_block(raw)
    lex_codes = lookup_disease_icd(diag_text)
    icd_codes = prioritize_codes(list(lex_codes or []))
    icd_from_kz = rs.extract_icd_codes_diagnosis_focused(raw)
    if icd_from_kz:
        icd_codes = prioritize_codes(list(icd_codes) + list(icd_from_kz))

    doctor_rubric = specialty_to_rubric(_detect_specialty(raw[:1500]) or _detect_specialty(raw))
    if doctor_rubric not in rs.ALLOWED_SPECIALTY_SLUGS:
        doctor_rubric = None
    user_slugs = [specialty_slug.strip()] if specialty_slug and specialty_slug in rs.ALLOWED_SPECIALTY_SLUGS else []
    target_slugs = list(dict.fromkeys(([doctor_rubric] if doctor_rubric else []) + user_slugs))

    demographics_meta = demographics_meta if isinstance(demographics_meta, dict) else {}
    clinical_rules = rs._consult_clinical_rules_pipeline(
        raw,
        demographics_meta,
        list(icd_codes),
        user_slugs,
    )
    allowed_paths, _meta = consult_target_protocol_paths(
        merged_icd=list(icd_codes),
        diag_icd=list(icd_codes),
        clinical_rules=clinical_rules if isinstance(clinical_rules, dict) else None,
        specialty_slugs=target_slugs or None,
        consult_text=raw,
        consult_facts=(
            clinical_rules.get("consult_facts") if isinstance(clinical_rules, dict) else None
        ),
        primary_specialty=doctor_rubric or None,
    )
    if not allowed_paths and isinstance(clinical_rules, dict):
        for mp in clinical_rules.get("matched_protocols") or []:
            sp = (mp or {}).get("source_path")
            if sp and sp not in allowed_paths:
                allowed_paths.append(sp)

    q_rag = rs.clinical_query_for_rag(raw) or raw[:7000]
    rq = raw[: min(len(raw), int(os.environ.get("PATIENT_RAG_QUERY_CHARS", "6000")))]
    embed_rerank = rs._consult_retrieve_embed_rerank()
    max_chunks = _patient_rag_max_chunks()
    max_per_path = max(1, min(3, int(os.environ.get("PATIENT_RAG_MAX_PER_PATH", "2"))))

    retrieved: list[dict] = []
    try:
        retrieved = rs.retrieve(
            q_rag,
            routing_query=rq,
            user_category_slugs=target_slugs or None,
            icd_codes_for_lex=list(icd_codes) or None,
            path_boost=allowed_paths or None,
            path_allowlist=allowed_paths or None,
            max_chunks=max_chunks,
            max_per_path=max_per_path,
            embed_rerank=embed_rerank,
        )
    except Exception:
        retrieved = []

    if not retrieved and icd_codes:
        try:
            retrieved = rs.retrieve(
                " ".join(icd_codes[:6]),
                routing_query=rq,
                user_category_slugs=target_slugs or None,
                icd_codes_for_lex=list(icd_codes),
                path_boost=allowed_paths or None,
                path_allowlist=allowed_paths or None,
                max_chunks=max_chunks,
                max_per_path=max_per_path,
                embed_rerank=embed_rerank,
            )
        except Exception:
            retrieved = []

    if allowed_paths and retrieved:
        retrieved = filter_retrieval_rows_by_paths(retrieved, allowed_paths)

    paths: list[str] = []
    seen: set[str] = set()
    for row in allowed_paths or []:
        p = str(row or "").replace("\\", "/").strip()
        if p and p not in seen:
            seen.add(p)
            paths.append(p)
    for row in retrieved or []:
        p = str(row.get("path") or "").replace("\\", "/").strip()
        if p and p not in seen:
            seen.add(p)
            paths.append(p)
        if len(paths) >= _patient_rag_max_paths():
            break

    vector_used = False
    try:
        from clinical_knowledge.vector_index import index_stats, vector_index_enabled

        vector_used = bool(vector_index_enabled() and index_stats().get("loaded"))
    except Exception:
        vector_used = False

    return {
        "paths": paths[: _patient_rag_max_paths()],
        "retrieved": list(retrieved or [])[:max_chunks],
        "icd_codes": list(icd_codes or [])[:8],
        "rag_used": bool(retrieved),
        "vector_used": vector_used,
    }


def patient_protocol_citations_from_retrieved(
    retrieved: list[dict[str, Any]],
    *,
    limit: int = 4,
) -> list[dict[str, str]]:
    """Короткие цитаты из corpus chunks для пациента."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in retrieved or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if len(text) < 40:
            continue
        fp = re.sub(r"\s+", " ", text.lower())[:100]
        if fp in seen:
            continue
        seen.add(fp)
        title = str(row.get("section_title") or row.get("title") or "Протокол").strip()[:120]
        out.append(
            {
                "title": title,
                "excerpt": text[:320].rstrip() + ("…" if len(text) > 320 else ""),
                "path": str(row.get("path") or ""),
            }
        )
        if len(out) >= limit:
            break
    return out
