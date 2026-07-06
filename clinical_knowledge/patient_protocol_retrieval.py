"""RAG-подбор протоколов для B2C: corpus chunks + vector embeddings (как в КЗ L2 и поиске)."""
from __future__ import annotations

import os
import re
from typing import Any

from .patient_flags import patient_rag_retrieval_enabled

_MEDICAL_KZ_HINT = re.compile(
    r"ж[aа]л[oо]б|д[iіaа]агноз|\bр\s*:|р[eе]к[oо0]менд|назнач|лечен|осмотр|контрол|"
    r"объективн|анамнез|заключен|консультац",
    re.I,
)


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


def _chunk_fingerprint(row: dict[str, Any]) -> str:
    text = str(row.get("text") or "").strip().lower()
    return re.sub(r"\s+", " ", text)[:100]


def merge_patient_rag_context(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Объединить два RAG-контекста без дублей paths/retrieved."""
    paths: list[str] = []
    seen_paths: set[str] = set()
    for group in (base.get("paths") or [], extra.get("paths") or []):
        for row in group:
            p = str(row or "").replace("\\", "/").strip()
            if p and p not in seen_paths:
                seen_paths.add(p)
                paths.append(p)
    retrieved: list[dict] = []
    seen_chunks: set[str] = set()
    for group in (base.get("retrieved") or [], extra.get("retrieved") or []):
        for row in group:
            if not isinstance(row, dict):
                continue
            fp = _chunk_fingerprint(row)
            if not fp or fp in seen_chunks:
                continue
            seen_chunks.add(fp)
            retrieved.append(row)
    max_paths = _patient_rag_max_paths()
    max_chunks = _patient_rag_max_chunks()
    icd = list(dict.fromkeys(list(base.get("icd_codes") or []) + list(extra.get("icd_codes") or [])))
    return {
        "paths": paths[:max_paths],
        "retrieved": retrieved[:max_chunks],
        "icd_codes": icd[:8],
        "rag_used": bool(retrieved),
        "vector_used": bool(base.get("vector_used")) or bool(extra.get("vector_used")),
        "semantic_primary": bool(base.get("semantic_primary")) or bool(extra.get("semantic_primary")),
    }


def rag_probe_looks_like_kz(probe: dict[str, Any], kz_text: str) -> bool:
    """Консервативный критерий: короткий/шумный текст похож на КЗ по RAG + лексике."""
    rows = [r for r in (probe.get("retrieved") or []) if isinstance(r, dict)]
    if len(rows) < 2:
        return False
    protocol_rows = 0
    for row in rows:
        path = str(row.get("path") or "").lower()
        text = str(row.get("text") or "").strip()
        if not (
            path.endswith(".pdf")
            or "minzdrav" in path
            or "protocol" in path
            or "corpus" in path
        ):
            continue
        if len(text) >= 40 or path:
            protocol_rows += 1
    if protocol_rows < 2:
        return False
    hints = len(_MEDICAL_KZ_HINT.findall(kz_text or ""))
    if hints >= 2:
        return True
    if len((kz_text or "").strip()) >= 40 and hints >= 1 and protocol_rows >= 2:
        return True
    return False


def retrieve_patient_protocol_context(
    *,
    kz_text: str,
    demographics_meta: dict[str, Any] | None = None,
    specialty_slug: str | None = None,
    semantic_primary: bool = False,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    """
    Подбор путей протоколов и top-чанков через retrieve() с embed-rerank.
    Не бросает исключений - при сбое возвращает пустой контекст.

    semantic_primary: без path_allowlist и без filter_retrieval_rows_by_paths.
    """
    empty: dict[str, Any] = {
        "paths": [],
        "retrieved": [],
        "icd_codes": [],
        "rag_used": False,
        "vector_used": False,
        "semantic_primary": semantic_primary,
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
        if not getattr(rs, "_chunks", None):
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
    chunk_limit = max_chunks if max_chunks is not None else _patient_rag_max_chunks()
    max_per_path = max(1, min(3, int(os.environ.get("PATIENT_RAG_MAX_PER_PATH", "2"))))

    path_boost = None if semantic_primary else (allowed_paths or None)
    path_allowlist = None if semantic_primary else (allowed_paths or None)

    retrieved: list[dict] = []
    try:
        retrieved = rs.retrieve(
            q_rag,
            routing_query=rq,
            user_category_slugs=target_slugs or None,
            icd_codes_for_lex=list(icd_codes) or None,
            path_boost=path_boost,
            path_allowlist=path_allowlist,
            max_chunks=chunk_limit,
            max_per_path=max_per_path,
            embed_rerank=embed_rerank,
        )
    except Exception:
        retrieved = []

    if not retrieved and icd_codes and not semantic_primary:
        try:
            retrieved = rs.retrieve(
                " ".join(icd_codes[:6]),
                routing_query=rq,
                user_category_slugs=target_slugs or None,
                icd_codes_for_lex=list(icd_codes),
                path_boost=allowed_paths or None,
                path_allowlist=allowed_paths or None,
                max_chunks=chunk_limit,
                max_per_path=max_per_path,
                embed_rerank=embed_rerank,
            )
        except Exception:
            retrieved = []

    if allowed_paths and retrieved and not semantic_primary:
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
        "retrieved": list(retrieved or [])[:chunk_limit],
        "icd_codes": list(icd_codes or [])[:8],
        "rag_used": bool(retrieved),
        "vector_used": vector_used,
        "semantic_primary": semantic_primary,
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
