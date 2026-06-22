"""Пайплайн проверки КЗ с поэтапным прогрессом (SSE)."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

ProgressFn = Any  # Callable[[str, int, str, dict], None] - optional legacy


def _progress_tuple(stage: str, pct: int, label_ru: str, partial: dict | None = None) -> tuple[str, dict]:
    return (
        "progress",
        {
            "stage": stage,
            "pct": pct,
            "label_ru": label_ru,
            "partial": partial or {},
        },
    )


def _protocol_rows_from_rich_paths(
    paths: list[str],
    *,
    query: str,
    icd_codes: list[str],
    get_chunks: Any,
    limit_per_path: int = 2,
    max_paths: int = 4,
) -> list[dict[str, Any]]:
    """Компактные фрагменты протоколов без retrieve() по всему корпусу."""
    from clinical_knowledge.protocol_practical_lite import _pick_chunks

    rows: list[dict[str, Any]] = []
    for path in paths[:max_paths]:
        p = str(path or "").strip()
        if not p:
            continue
        chunks = get_chunks(p) or []
        if not chunks:
            continue
        try:
            from clinical_knowledge.chunk_tags import chunk_usable_for_retrieval

            chunks = [c for c in chunks if chunk_usable_for_retrieval(c, ambulatory=True)]
        except Exception:
            pass
        picked = _pick_chunks(chunks, query, icd_codes, limit=limit_per_path)
        for ch in picked:
            txt = (ch.get("text") or ch.get("lex_text") or "").strip()
            if len(txt) < 40:
                continue
            rows.append(
                {
                    "path": p,
                    "text": txt,
                    "excerpt": txt[:2000],
                    "kind": ch.get("kind") or "",
                    "section_title": ch.get("section_title") or "",
                    "page_from": ch.get("page_from"),
                    "page_to": ch.get("page_to"),
                }
            )
    return rows


def _iter_consult_review_render_l2_lite(
    *,
    full_text: str,
    n_files: int,
    consult_docs_meta: list[dict],
    pdf_warnings: list[str],
    content_signature: str,
    category_slugs: str,
    fhir_bundle: dict[str, Any] | None,
    cache_key: str,
    emit: Any,
) -> Iterator[tuple[str, Any]]:
    """Render-safe L2: L1 structured + rich-чанки по matched paths + LLM (без retrieve по 65k)."""
    import rag_server as rs
    from clinical_knowledge.consult_parser import _detect_specialty
    from clinical_knowledge.consult_tiering import run_l1_structured_review
    from clinical_knowledge.rubric_extractors import specialty_to_rubric

    yield emit("focus", 18, "Структурный разбор заключения…")
    doctor_rubric = specialty_to_rubric(_detect_specialty(full_text[:1500]) or _detect_specialty(full_text))
    user_slugs = [
        s.strip()
        for s in (category_slugs or "").split(",")
        if s.strip() in rs.ALLOWED_SPECIALTY_SLUGS
    ]
    specialty_slug = (
        doctor_rubric
        if doctor_rubric in rs.ALLOWED_SPECIALTY_SLUGS
        else (user_slugs[0] if len(user_slugs) == 1 else None)
    )
    demographics_banner, demographics_meta = rs.consult_demographics_banner_from_kz(full_text)
    consult_id = (content_signature or "consult")[:16] or "consult"

    l1 = run_l1_structured_review(
        text=full_text,
        consultation_id=consult_id,
        demographics_meta=demographics_meta if isinstance(demographics_meta, dict) else None,
        specialty_slug=specialty_slug,
    )
    structured_analysis = l1.get("structured_analysis")
    alignment_result = l1.get("alignment")
    doc = None
    if isinstance(structured_analysis, dict):
        doc = structured_analysis.get("document")

    icd_codes: list[str] = []
    if doc is not None and getattr(doc, "diagnoses", None):
        icd_codes = [
            str(d.icd10_code).upper()
            for d in doc.diagnoses
            if getattr(d, "icd10_code", None)
        ]
    if not icd_codes:
        icd_codes = rs.extract_icd_codes_diagnosis_focused(full_text) or []

    matches = (structured_analysis or {}).get("matches") if isinstance(structured_analysis, dict) else []
    match_paths = [
        str(m.get("source_path") or "")
        for m in (matches or [])
        if isinstance(m, dict) and m.get("source_path")
    ]
    if alignment_result and isinstance(alignment_result.get("protocol_paths"), list):
        for p in alignment_result["protocol_paths"]:
            ps = str(p or "").strip()
            if ps and ps not in match_paths:
                match_paths.append(ps)

    yield emit(
        "protocols",
        52,
        f"Фрагменты протоколов ({len(match_paths[:4])})…",
        {"protocol_paths_target": match_paths[:6]},
    )
    q_rag = " ".join(icd_codes[:6]) or full_text[:1200]
    retrieved = _protocol_rows_from_rich_paths(
        match_paths,
        query=q_rag,
        icd_codes=icd_codes,
        get_chunks=rs.get_rich_chunks_for_path,
        limit_per_path=2,
        max_paths=4,
    )
    proto_max = rs._consult_env_int("CONSULT_REVIEW_PROTOCOL_CTX_CHARS", 16500, default_fast=8000)
    protocol_ctx, paths_used = rs._build_review_chunks_context(retrieved, proto_max)
    paths_hint = rs._consult_review_paths_hint(paths_used, retrieved=retrieved, icd_needles=icd_codes[:6])
    ui_frags = rs._consult_ui_protocol_fragments(retrieved, paths_used)
    oncology = rs._consult_oncology_flags(ui_frags, full_text)

    clinical_rules = rs._consult_clinical_rules_pipeline(
        full_text,
        demographics_meta if isinstance(demographics_meta, dict) else {},
        icd_codes,
        user_slugs,
    )
    rules_ctx = ""
    if clinical_rules:
        try:
            from clinical_knowledge.llm_context import format_clinical_rules_for_llm

            rules_ctx = format_clinical_rules_for_llm(clinical_rules)
        except ImportError:
            rules_ctx = ""

    consult_max = rs._consult_env_int("CONSULT_REVIEW_CONSULT_CHARS", 20000, default_fast=12000)
    multi_intro = (
        ""
        if n_files <= 1
        else (
            "Несколько документов: блоки ниже - в порядке загрузки; при оценке учитывай "
            "согласованность между приёмами.\n\n"
        )
    )
    consult_excerpt = multi_intro + full_text[:consult_max]
    if demographics_banner.strip():
        consult_excerpt = demographics_banner.strip() + "\n\n" + consult_excerpt

    yield emit("synthesize", 78, "Формирование оценки (модель)…")
    model = rs.get_gemini()
    try:
        review = rs._consult_review_synthesize(
            model,
            consult_excerpt,
            protocol_ctx,
            paths_hint,
            clinical_rules_context=rules_ctx,
        )
    except Exception as exc:
        from fastapi import HTTPException

        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка финальной оценки модели: {str(exc)[:200]}",
        ) from exc

    if isinstance(alignment_result, dict):
        try:
            from clinical_knowledge.consult_alignment import merge_alignment_into_review

            merge_alignment_into_review(review, alignment_result)
        except Exception:
            pass

    if isinstance(review, dict):
        try:
            from clinical_knowledge.consult_overall_score import apply_hybrid_overall_compliance

            apply_hybrid_overall_compliance(
                review,
                structured_analysis=structured_analysis if isinstance(structured_analysis, dict) else None,
                clinical_rules=clinical_rules if isinstance(clinical_rules, dict) else None,
            )
        except Exception:
            pass

    result: dict[str, Any] = {
        "ok": True,
        "server_version": rs._app_version(),
        "review_tier": "L2",
        "review": review,
        "pdf_warnings": pdf_warnings,
        "consult_documents": consult_docs_meta,
        "documents_count": len(consult_docs_meta),
        "extraction_chars": len(full_text),
        "retrieval_paths": paths_used,
        "consult_protocol_fragments": ui_frags,
        "consult_oncology_flags": oncology,
        "consult_icd_precise_links": [],
        "consult_icd_precise_note_ru": "",
        "audience_filter": None,
        "audience_fallback": False,
        "consult_retrieval": {
            "render_l2_lite": True,
            "strict_protocol_paths": match_paths[:6],
            "path_pick_meta": {"source": "l1_matches"},
        },
        "icd": {"codes": icd_codes[:12]},
        "demographics_meta": demographics_meta,
        "consult_performance": {
            "fast_mode": True,
            "render_l2_lite": True,
            "embed_rerank": False,
            "rag_retrieve_skipped": True,
        },
    }
    if clinical_rules is not None:
        result["clinical_rules"] = clinical_rules
    if alignment_result is not None:
        result["alignment"] = alignment_result
    if structured_analysis is not None:
        result["structured_analysis"] = structured_analysis

    if isinstance(structured_analysis, dict):
        comp = structured_analysis.get("compliance")
        if isinstance(comp, dict):
            try:
                from clinical_knowledge.compliance_gate import evaluate_send_gate_from_compliance

                headline = review.get("overall_compliance_pct") if isinstance(review, dict) else None
                hs = float(headline) if isinstance(headline, (int, float)) else None
                sg = evaluate_send_gate_from_compliance(comp, headline_score=hs)
                comp["send_gate"] = sg
                result["send_gate"] = sg
            except Exception:
                pass

    try:
        from clinical_knowledge.cisz_readiness import attach_cisz_readiness

        attach_cisz_readiness(
            result,
            bundle=fhir_bundle,
            text=full_text if not fhir_bundle else None,
        )
    except Exception:
        pass

    result["cached_result"] = False
    rs._consult_cache_put(cache_key, result)
    import gc

    gc.collect()
    yield ("done", result)


def iter_consult_review_pipeline(
    *,
    full_text: str,
    n_files: int,
    consult_docs_meta: list[dict],
    pdf_warnings: list[str],
    content_signature: str,
    category_slugs: str,
    fhir_bundle: dict[str, Any] | None = None,
    on_progress: ProgressFn | None = None,
) -> Iterator[tuple[str, Any]]:
    """Генератор: ('progress', {stage,pct,label_ru,partial}) | ('done', result dict)."""
    import rag_server as rs

    def emit(stage: str, pct: int, label_ru: str, partial: dict | None = None) -> tuple[str, dict]:
        ev = _progress_tuple(stage, pct, label_ru, partial)
        if on_progress:
            on_progress(stage, pct, label_ru, partial or {})
        return ev

    yield emit("cache", 8, "Проверка кэша…", {"consult_documents": consult_docs_meta})
    cache_key = rs._consult_cache_key(content_signature, category_slugs)
    cached = rs._consult_cache_get(cache_key)
    if cached is not None:
        yield emit("cache_hit", 100, "Результат из кэша", {"cached_result": True})
        try:
            from clinical_knowledge.cisz_readiness import attach_cisz_readiness

            attach_cisz_readiness(
                cached,
                bundle=fhir_bundle,
                text=full_text if not fhir_bundle else None,
            )
        except Exception:
            pass
        yield ("done", cached)
        return

    if rs._consult_render_l2_lite_enabled():
        yield from _iter_consult_review_render_l2_lite(
            full_text=full_text,
            n_files=n_files,
            consult_docs_meta=consult_docs_meta,
            pdf_warnings=pdf_warnings,
            content_signature=content_signature,
            category_slugs=category_slugs,
            fhir_bundle=fhir_bundle,
            cache_key=cache_key,
            emit=emit,
        )
        return

    yield emit("focus", 15, "Анализ текста заключения (фокус запроса)…")
    model = rs.get_gemini()
    synthetic, retrieval_focus_meta = rs._build_consult_review_pipeline_query(model, full_text)

    fast = rs._consult_review_fast_mode()
    icd_from_kz = bool(rs.extract_icd_codes_diagnosis_focused(full_text))

    yield emit("icd", 28, "Подбор кодов МКБ-10 и клинического контекста…")
    icd_analysis, q, q_rag, _, icd_err = rs._infer_icd_pipeline_from_full_query(
        synthetic,
        model,
        skip_query_refine=fast,
        skip_icd_gemini=fast or icd_from_kz,
    )
    q_slice_fb = int(__import__("os").environ.get("CONSULT_REVIEW_RAG_QUERY_CHARS", "9000"))
    fallback_synthetic_legacy = (
        "=== Жалобы и вопрос ===\n\n" + full_text[: min(len(full_text), q_slice_fb)]
    )
    if icd_err or not (q_rag or "").strip():
        if fast:
            q = synthetic.strip()
            q_rag = rs.clinical_query_for_rag(synthetic) or full_text[:7000]
            icd_analysis = rs.analyze_query_for_icd(q, q_rag)
        else:
            icd_analysis_fb, q_fb, q_rag_fb, _, icd_err_fb = rs._infer_icd_pipeline_from_full_query(
                fallback_synthetic_legacy, model
            )
            if not icd_err_fb and (q_rag_fb or "").strip():
                icd_analysis, q, q_rag = icd_analysis_fb, q_fb, q_rag_fb
            else:
                q = synthetic.strip()
                q_rag = rs.clinical_query_for_rag(synthetic) or full_text[:7000]
                icd_analysis = rs.analyze_query_for_icd(q, q_rag)

    merged_icd, icd_merge_meta = rs._merge_icd_codes_for_consult_retrieval(icd_analysis, full_text)
    diag_codes_list = (
        icd_merge_meta.get("diag_block_icd_codes")
        if isinstance(icd_merge_meta.get("diag_block_icd_codes"), list)
        else []
    )

    # --- Якорь по диагнозу/специальности КЗ (точный подбор протоколов) ---
    from clinical_knowledge.consult_parser import _detect_specialty
    from clinical_knowledge.diagnosis_icd import lookup_disease_icd, prioritize_codes
    from clinical_knowledge.rubric_extractors import specialty_to_rubric

    # Код болезни из словаря нозологий (по блоку «Диагноз», иначе по всему тексту),
    # чтобы симптом-код (R21.9 «сыпь») не уводил подбор КП в чужую рубрику.
    import re as _re

    _mdiag = _re.search(
        r"диагноз[^:\n]*[:\-]\s*(.+?)(?:\n\s*\n|рекомендац|обследован|лечени|назначен|$)",
        full_text,
        _re.I | _re.S,
    )
    diag_block_text = (_mdiag.group(1) if _mdiag else "") or full_text
    lex_codes = lookup_disease_icd(diag_block_text)
    if lex_codes:
        diag_codes_list = prioritize_codes(list(diag_codes_list) + lex_codes)
        merged_icd = prioritize_codes(list(merged_icd or []) + lex_codes)

    # Специальность врача из шапки КЗ - авторитетный якорь рубрики.
    doctor_specialty_kz = _detect_specialty(full_text[:1500]) or _detect_specialty(full_text)
    doctor_rubric = specialty_to_rubric(doctor_specialty_kz)
    if doctor_rubric not in rs.ALLOWED_SPECIALTY_SLUGS:
        doctor_rubric = None

    user_slugs = [
        s.strip()
        for s in (category_slugs or "").split(",")
        if s.strip() in rs.ALLOWED_SPECIALTY_SLUGS
    ]

    query_specialties: list[str] = []
    try:
        query_specialties = rs.infer_specialties_gemini(q, model) if q_rag.strip() else []
    except Exception:
        query_specialties = []
    # Рубрика врача КЗ - впереди (приоритетный якорь), затем угаданные/выбранные.
    boost_merged = list(
        dict.fromkeys(
            ([doctor_rubric] if doctor_rubric else [])
            + (query_specialties or [])
            + user_slugs
        )
    )
    # Для ретривала: штраф чанкам вне рубрики врача КЗ (мягкий фильтр), чтобы
    # глобальный фолбэк не уходил в чужую специальность.
    retrieval_category_slugs = list(
        dict.fromkeys((user_slugs or []) + ([doctor_rubric] if doctor_rubric else []))
    )

    rq = q
    demographics_banner, demographics_meta = rs.consult_demographics_banner_from_kz(full_text)
    prefix_parts: list[str] = []
    if demographics_banner.strip():
        prefix_parts.append(demographics_banner.strip())
    icd_banner = rs._consult_icd_banner_for_retrieval(list(diag_codes_list), merged_icd)
    if icd_banner.strip():
        prefix_parts.append(icd_banner.strip())
    if prefix_parts:
        head = "\n\n".join(prefix_parts) + "\n\n"
        q = head + q.lstrip()
        rq = head + rq.lstrip()
        qr_lim = max(900, int(__import__("os").environ.get("CONSULT_REVIEW_RAG_QUERY_CHARS", "9000")))
        q_rag = (head.strip() + "\n\n" + q_rag.strip()).strip()[:qr_lim]

    yield emit(
        "icd_done",
        38,
        "МКБ и демография готовы",
        {
            "icd_codes": (merged_icd or [])[:8],
            "icd_count": len(merged_icd or []),
        },
    )

    yield emit("rules", 45, "Проверка по правилам протоколов…")
    clinical_rules = rs._consult_clinical_rules_pipeline(
        full_text,
        demographics_meta if isinstance(demographics_meta, dict) else {},
        list(merged_icd or []),
        user_slugs,
    )
    from clinical_knowledge.consult_retrieval import (
        consult_target_protocol_paths,
        filter_retrieval_by_category_slugs,
        filter_retrieval_rows_by_paths,
    )

    strict_proto = rs.env_bool("CONSULT_REVIEW_STRICT_PROTOCOLS", True)
    # Скоуп кандидатов. Специальность врача КЗ авторитетна: если она распознана,
    # ограничиваем рубрику ею (+ явно выбранными пользователем), не доверяя
    # «угадыванию» рубрики по шумному запросу (источник ложного гастро-подбора).
    if doctor_rubric:
        target_slugs = list(dict.fromkeys([doctor_rubric, *(user_slugs or [])]))
    else:
        target_slugs = list(dict.fromkeys((boost_merged or []) + (user_slugs or [])))
    allowed_paths, path_pick_meta = consult_target_protocol_paths(
        merged_icd=list(merged_icd or []),
        diag_icd=list(diag_codes_list or []),
        clinical_rules=clinical_rules if isinstance(clinical_rules, dict) else None,
        specialty_slugs=target_slugs or None,
        consult_text=full_text,
        consult_facts=(
            clinical_rules.get("consult_facts")
            if isinstance(clinical_rules, dict)
            else None
        ),
        primary_specialty=doctor_rubric or None,
    )
    matched_path_boost = list(allowed_paths)
    if not matched_path_boost and isinstance(clinical_rules, dict):
        for mp in clinical_rules.get("matched_protocols") or []:
            sp = (mp or {}).get("source_path")
            if sp and sp not in matched_path_boost:
                matched_path_boost.append(sp)

    rules_partial: dict[str, Any] = {
        "matched_protocols_count": len(matched_path_boost),
        "icd_codes": (merged_icd or [])[:6],
    }
    if isinstance(clinical_rules, dict):
        rc = clinical_rules.get("rules_check") or {}
        if isinstance(rc, dict) and rc.get("rules_compliance_pct") is not None:
            rules_partial["rules_compliance_pct"] = rc.get("rules_compliance_pct")
    yield emit("rules_done", 52, "Правила и протоколы определены", rules_partial)

    if rs.env_bool("RENDER", False):
        import gc

        gc.collect()

    max_chunks_r = rs._consult_env_int("CONSULT_REVIEW_MAX_CHUNKS", 12, default_fast=8)
    max_per_path_r = rs._consult_env_int("CONSULT_REVIEW_MAX_PER_PATH", 3, default_fast=2)
    embed_rerank = rs._consult_retrieve_embed_rerank()
    path_allow = matched_path_boost if strict_proto and matched_path_boost else None
    icd_for_retrieval = list(diag_codes_list or merged_icd or [])

    yield emit(
        "retrieve",
        58,
        f"Поиск по {len(path_allow or matched_path_boost or []) or 'релевантным'} протоколам…",
        {"protocol_paths_target": (path_allow or matched_path_boost)[:6]},
    )
    retrieved = rs.retrieve(
        q_rag,
        routing_query=rq,
        category_boost=boost_merged or None,
        user_category_slugs=retrieval_category_slugs or None,
        icd_codes_for_lex=icd_for_retrieval or None,
        path_boost=matched_path_boost or None,
        path_allowlist=path_allow,
        max_chunks=max_chunks_r,
        max_per_path=max_per_path_r,
        embed_rerank=embed_rerank,
    )
    if not retrieved and path_allow:
        retrieved = rs.retrieve(
            q_rag,
            routing_query=rq,
            category_boost=boost_merged or None,
            user_category_slugs=retrieval_category_slugs or None,
            icd_codes_for_lex=icd_for_retrieval or None,
            path_boost=matched_path_boost or None,
            path_allowlist=None,
            max_chunks=max_chunks_r,
            max_per_path=max_per_path_r,
            embed_rerank=embed_rerank,
        )
    if not retrieved and icd_for_retrieval:
        retrieved = rs.retrieve(
            " ".join(icd_for_retrieval[:6]),
            routing_query=rq,
            category_boost=boost_merged or None,
            user_category_slugs=retrieval_category_slugs or None,
            icd_codes_for_lex=icd_for_retrieval,
            path_boost=matched_path_boost or None,
            path_allowlist=path_allow,
            max_chunks=max_chunks_r,
            max_per_path=max_per_path_r,
            embed_rerank=embed_rerank,
        )
    if not retrieved:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=(
                "Не удалось подобрать фрагменты протоколов по МКБ и диагнозу из КЗ. "
                "Укажите коды МКБ-10 в документе или выберите рубрику протокола."
            ),
        )

    retrieved = filter_retrieval_rows_by_paths(retrieved, path_allow)
    if path_allow and not retrieved:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail="По выбранным протоколам и кодам МКБ не найдено релевантных фрагментов. Проверьте коды МКБ в КЗ.",
        )

    second_pass_on = rs._consult_rag_second_pass_enabled() and not (strict_proto and path_allow)
    second_pass_diag: dict = {
        "enabled": second_pass_on,
        "applied": False,
        "reason": "",
        "trigger_eval": False,
    }

    if retrieved:
        m1 = rs._consult_retrieval_quality_metrics(retrieved)
        second_pass_diag["first_pass_metrics"] = {
            "max_score": round(float(m1["max_score"]), 4),
            "top3_lex_avg": round(float(m1["top3_lex_avg"]), 4),
            "n_chunks": int(m1["n_chunks"]),
            "uniq_paths": int(m1["uniq_paths"]),
        }
        need2 = False
        why = ""
        if second_pass_on:
            need2, why = rs._consult_should_second_pass(m1)
        second_pass_diag["trigger_eval"] = bool(need2)
        second_pass_diag["trigger_reason_code"] = why if second_pass_on else "second_pass_disabled"
        if second_pass_on and need2:
            yield emit("retrieve2", 65, "Уточняющий поиск по протоколам…")
            try:
                q2, aug_meta = rs._consult_second_pass_build_query(
                    model,
                    q_rag,
                    rq,
                    retrieval_focus_meta if isinstance(retrieval_focus_meta, dict) else {},
                    retrieved,
                )
                second_pass_diag["augment"] = aug_meta
                bump = max(0, int(__import__("os").environ.get("CONSULT_REVIEW_RAG_SECOND_PASS_EXTRA_CHUNKS", "4")))
                r2 = rs.retrieve(
                    q2.strip(),
                    routing_query=rq,
                    category_boost=boost_merged or None,
                    user_category_slugs=retrieval_category_slugs or None,
                    icd_codes_for_lex=icd_for_retrieval,
                    path_boost=matched_path_boost or None,
                    path_allowlist=path_allow,
                    max_chunks=max_chunks_r + bump,
                    max_per_path=max_per_path_r,
                    embed_rerank=embed_rerank,
                )
                if r2:
                    merge_max = max_chunks_r + bump
                    retrieved = rs._merge_chunk_retrieval_lists(
                        [retrieved, r2],
                        max_chunks=merge_max,
                        max_per_path=max_per_path_r,
                    )
                    retrieved = filter_retrieval_rows_by_paths(retrieved, path_allow)
                    second_pass_diag["applied"] = True
                    second_pass_diag["reason"] = why
                    second_pass_diag["second_retrieve_rows"] = len(r2)
                    second_pass_diag["merged_rows"] = len(retrieved)
                else:
                    second_pass_diag["reason"] = (
                        why + ";second_retrieve_empty" if why else "second_retrieve_empty"
                    )
            except Exception as e:
                second_pass_diag["reason"] = (why + ";exception") if why else "exception"
                second_pass_diag["error"] = str(e)[:240]
        elif not second_pass_on:
            second_pass_diag["reason"] = "feature_disabled_env"
        else:
            second_pass_diag["reason"] = "first_pass_ok"

    retrieved, audience_hint, audience_fb = rs.filter_retrieval_by_audience(retrieved, rq, rs._routing)

    if doctor_rubric and rs.env_bool("CONSULT_REVIEW_STRICT_SPECIALTY", True):
        specialty_slugs = list(dict.fromkeys([doctor_rubric] + (user_slugs or [])))
        retrieved = filter_retrieval_by_category_slugs(
            retrieved,
            specialty_slugs,
            strict=True,
        )

    _acoag_markers = (
        "ривароксабан", "апиксабан", "варфарин", "дабигатран", "эноксапарин",
        "антикоагул", "флеботромбоз", "тромбоз глубок", "тгв",
    )
    if any(m in full_text.lower() for m in _acoag_markers):
        try:
            med_q = (
                "антикоагулянтная терапия дозировка длительность ривароксабан "
                "прямые оральные антикоагулянты контроль узи глубоких вен"
            )
            r_med = rs.retrieve(
                med_q,
                routing_query=rq,
                category_boost=boost_merged or None,
                user_category_slugs=retrieval_category_slugs or None,
                icd_codes_for_lex=icd_for_retrieval,
                path_boost=matched_path_boost or None,
                path_allowlist=path_allow,
                max_chunks=4,
                max_per_path=2,
                embed_rerank=embed_rerank,
            )
            if r_med:
                retrieved = rs._merge_chunk_retrieval_lists(
                    [retrieved, r_med],
                    max_chunks=max_chunks_r + 4,
                    max_per_path=max_per_path_r,
                )
                retrieved = filter_retrieval_rows_by_paths(retrieved, path_allow)
        except Exception:
            pass

    if rs.env_bool("CONSULT_TYPED_RETRIEVE", True):
        try:
            from clinical_knowledge.consult_retrieval import supplement_retrieval_from_rich_chunks

            retrieved = supplement_retrieval_from_rich_chunks(
                retrieved,
                paths=list(path_allow or matched_path_boost or []),
                icd_codes=list(icd_for_retrieval or []),
                get_chunks=rs.get_rich_chunks_for_path,
                query=q_rag,
            )
        except Exception:
            pass

    icd_frag_needles = rs._consult_needles_icd_fragments_consult_review(list(diag_codes_list), merged_icd)
    retrieved = rs._consult_sort_retrieval_by_icd_fragments_first(retrieved, icd_frag_needles)
    precise_links, precise_note_ru = rs._consult_precise_links_for_icd_in_fragments(
        retrieved,
        diag_block_icd=list(diag_codes_list),
        merged_icd=merged_icd,
    )

    proto_max = rs._consult_env_int("CONSULT_REVIEW_PROTOCOL_CTX_CHARS", 16500, default_fast=11000)
    protocol_ctx, paths_used = rs._build_review_chunks_context(retrieved, proto_max)
    paths_hint_for_llm = rs._consult_review_paths_hint(
        paths_used,
        retrieved=retrieved,
        icd_needles=icd_frag_needles,
    )

    ui_frags = rs._consult_ui_protocol_fragments(retrieved, paths_used)
    oncology = rs._consult_oncology_flags(ui_frags, full_text)

    yield emit(
        "protocols",
        72,
        f"Подобрано протоколов: {len(paths_used)}",
        {
            "retrieval_paths_count": len(paths_used),
            "protocol_paths_used": paths_used[:8],
        },
    )

    # Summary-fallback по МКБ: после RAG (не блокирует поиск), ~0.1 с вместо ~30 с на полный корпус.
    try:
        from clinical_knowledge.rules_summary_fallback import apply_summary_rules_fallback

        clinical_rules = apply_summary_rules_fallback(clinical_rules, list(merged_icd or []))
        if isinstance(clinical_rules, dict):
            rc = clinical_rules.get("rules_check") or {}
            if isinstance(rc, dict) and rc.get("summary_fallback_applied"):
                rules_partial["rules_compliance_pct"] = rc.get("rules_compliance_pct")
                rules_partial["summary_fallback_applied"] = True
    except Exception:
        pass

    consult_max = rs._consult_env_int("CONSULT_REVIEW_CONSULT_CHARS", 20000, default_fast=14000)
    multi_intro = (
        ""
        if n_files <= 1
        else (
            "Несколько документов: блоки ниже - в порядке загрузки; при оценке учитывай "
            "согласованность между приёмами, хронологию формулировок и возможные противоречия между частями.\n\n"
        )
    )
    reserve_for_suffix = 100
    room = max(400, consult_max - len(multi_intro) - reserve_for_suffix)
    consult_body = full_text[:room].strip()
    suffix = ""
    if len(full_text) > len(consult_body):
        suffix += "\n\n[…остаток заключений не передан в модель из-за лимита]"
    consult_excerpt = multi_intro + consult_body + suffix

    oncology_extra = ""
    if oncology.get("any"):
        oncology_extra = (
            "\n\nВАЖНО ДЛЯ ОЦЕНКИ ЭТОГО КОНСУЛЬТАТИВНОГО ЗАКЛЮЧЕНИЯ:\n"
            + str(oncology.get("instruction_ru") or "").strip()
            + "\nЕсли текст заключения связан с онкологическим риском или опухолевой патологией, отдельно оцените клиническую "
            "безопасность формулировок применительно к переданным фрагментам протоколов; при недостаточном покрытии протоколами усильте ограничения "
            "(limitations_ru) и понизьте баллы по затронутым критериям.\n"
        )

    try:
        from clinical_knowledge.llm_context import format_clinical_rules_for_llm

        rules_ctx = format_clinical_rules_for_llm(clinical_rules)
    except ImportError:
        rules_ctx = ""

    yield emit("synthesize", 85, "Формирование оценки (модель)…")
    try:
        review = rs._consult_review_synthesize(
            model,
            consult_excerpt,
            protocol_ctx,
            paths_hint_for_llm,
            extra_context=oncology_extra,
            clinical_rules_context=rules_ctx,
        )
    except Exception as exc:
        from fastapi import HTTPException

        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка финальной оценки модели: {str(exc)[:200]}",
        ) from exc

    # --- Единый parse КЗ для structured + alignment ---
    parsed_doc = None
    try:
        from clinical_knowledge.consult_parser import parse_consultation

        parsed_doc = parse_consultation(
            full_text,
            consultation_id=(content_signature or "consult")[:16] or "consult",
            demographics_meta=demographics_meta if isinstance(demographics_meta, dict) else None,
        )
    except Exception:
        parsed_doc = None

    # --- Структурный детерминированный разбор КЗ (аддитивно, за флагом) ---
    structured_analysis = None
    report_markdown = None
    report_html = None
    if rs.env_bool("CONSULT_STRUCTURED_ANALYSIS", True):
        try:
            from clinical_knowledge.consult_analysis import analyze_consultation_text

            yield emit("structured", 92, "Структурный разбор заключения…")
            sa = analyze_consultation_text(
                full_text,
                consultation_id=(content_signature or "consult")[:16] or "consult",
                demographics_meta=demographics_meta if isinstance(demographics_meta, dict) else None,
                specialty_slug=doctor_rubric,
                with_markdown=rs._consult_response_include_html(),
                doc=parsed_doc,
                analysis_mode=(
                    os.environ.get("PROTOCOL_SUMMARY_MODE")
                    if rs.env_bool("PROTOCOL_SUMMARY_ENABLED", False)
                    else "legacy"
                ),
            )
            structured_analysis = {
                "document": sa.get("document"),
                "matches": sa.get("matches"),
                "compliance": sa.get("compliance"),
                "rubric_specifics": sa.get("rubric_specifics"),
            }
            report_markdown = sa.get("report_markdown")
            report_html = sa.get("report_html")
        except Exception:
            structured_analysis = None

    # --- Детерминированные карточки согласования (МКБ / КП / НПА) ---
    alignment_result = None
    if rs.env_bool("CONSULT_ALIGNMENT_ENABLED", True):
        try:
            from clinical_knowledge.consult_alignment import (
                append_alignment_evidence,
                build_consult_alignment,
                merge_alignment_into_review,
                sync_structured_with_alignment,
            )
            from clinical_knowledge.consult_parser import parse_consultation
            from clinical_knowledge.consult_retrieval import unify_consult_protocol_paths

            if parsed_doc is None:
                parsed_doc = parse_consultation(
                    full_text,
                    consultation_id=(content_signature or "consult")[:16] or "consult",
                    demographics_meta=demographics_meta if isinstance(demographics_meta, dict) else None,
                )
            rules_paths = [
                str((mp or {}).get("source_path") or "")
                for mp in ((clinical_rules or {}).get("matched_protocols") or [])
                if isinstance(mp, dict) and mp.get("source_path")
            ]
            alignment_paths = unify_consult_protocol_paths(
                target_paths=list(matched_path_boost or []),
                rules_paths=rules_paths,
                rag_paths=list(paths_used or []),
            )
            alignment_result = build_consult_alignment(
                parsed_doc,
                protocol_paths=alignment_paths,
                icd_codes=list(merged_icd or diag_codes_list or []),
                get_chunks=rs.get_rich_chunks_for_path,
                query=q_rag or q,
                protocol_matches=(
                    path_pick_meta.get("protocol_matches")
                    if isinstance(path_pick_meta, dict)
                    else None
                ),
                specialty_slug=doctor_rubric or None,
                specialty_label=parsed_doc.doctor_specialty if parsed_doc else None,
            )
            if isinstance(review, dict):
                merge_alignment_into_review(review, alignment_result)
            sync_structured_with_alignment(structured_analysis, alignment_result)
            append_alignment_evidence(structured_analysis, alignment_result)
        except Exception:
            alignment_result = None

    if isinstance(review, dict):
        try:
            from clinical_knowledge.consult_overall_score import apply_hybrid_overall_compliance

            apply_hybrid_overall_compliance(
                review,
                structured_analysis=structured_analysis,
                clinical_rules=clinical_rules if isinstance(clinical_rules, dict) else None,
            )
        except Exception:
            pass

    result = {
        "ok": True,
        "server_version": rs._app_version(),
        "review": review,
        "pdf_warnings": pdf_warnings,
        "consult_documents": consult_docs_meta,
        "documents_count": len(consult_docs_meta),
        "extraction_chars": len(full_text),
        "retrieval_paths": paths_used,
        "consult_protocol_fragments": ui_frags,
        "consult_oncology_flags": oncology,
        "consult_icd_precise_links": precise_links,
        "consult_icd_precise_note_ru": precise_note_ru,
        "audience_filter": audience_hint,
        "audience_fallback": audience_fb,
        "consult_retrieval": {
            "focus": retrieval_focus_meta,
            "icd_codes_lex_merged": merged_icd,
            "diag_block_icd_codes": diag_codes_list,
            "icd_merge_meta": icd_merge_meta,
            "fragments_icd_needles": icd_frag_needles,
            "second_pass": second_pass_diag,
            "strict_protocol_paths": path_allow or matched_path_boost,
            "path_pick_meta": path_pick_meta,
        },
        "icd": rs._icd_client_payload(icd_analysis),
        "demographics_meta": demographics_meta,
        "consult_performance": {
            "fast_mode": fast,
            "embed_rerank": embed_rerank,
            "icd_from_kz_text": icd_from_kz,
        },
    }
    if clinical_rules is not None:
        result["clinical_rules"] = clinical_rules
    if alignment_result is not None:
        result["alignment"] = alignment_result
    if structured_analysis is not None:
        result["structured_analysis"] = structured_analysis
    if report_markdown:
        result["report_markdown"] = report_markdown

    if isinstance(structured_analysis, dict):
        comp = structured_analysis.get("compliance")
        if isinstance(comp, dict):
            try:
                from clinical_knowledge.compliance_gate import evaluate_send_gate_from_compliance

                headline = (
                    review.get("overall_compliance_pct")
                    if isinstance(review, dict)
                    else None
                )
                hs = float(headline) if isinstance(headline, (int, float)) else None
                sg = evaluate_send_gate_from_compliance(comp, headline_score=hs)
                comp["send_gate"] = sg
                result["send_gate"] = sg
                if report_html:
                    from clinical_knowledge.consult_report import patch_report_html_send_gate

                    report_html = patch_report_html_send_gate(report_html, sg)
            except Exception:
                pass

    if report_html:
        result["report_html"] = report_html

    try:
        from clinical_knowledge.cisz_readiness import attach_cisz_readiness

        attach_cisz_readiness(
            result,
            bundle=fhir_bundle,
            text=full_text if not fhir_bundle else None,
        )
    except Exception:
        pass

    # Опционально: тихо дописать снимок в manifest на диске (CONSULT_ARCHIVE_ANALYSES=1).
    try:
        from clinical_knowledge.analysis_archive import build_snapshot, save_snapshot

        src_name = ""
        if consult_docs_meta and isinstance(consult_docs_meta[0], dict):
            src_name = str(consult_docs_meta[0].get("filename") or "")
        snap = build_snapshot(
            full_text=full_text,
            source_file=src_name,
            build_version=rs._app_version(),
            structured_analysis=structured_analysis,
            review=review if isinstance(review, dict) else None,
            retrieval_paths=paths_used,
            icd_codes=merged_icd,
        )
        save_snapshot(snap)
    except Exception:
        pass

    result["cached_result"] = False
    if not rs._consult_response_include_html():
        result.pop("report_html", None)
        result.pop("report_markdown", None)
    rs._consult_cache_put(cache_key, result)
    import gc

    gc.collect()
    yield ("done", result)


def run_consult_review_pipeline(
    *,
    full_text: str,
    n_files: int,
    consult_docs_meta: list[dict],
    pdf_warnings: list[str],
    content_signature: str,
    category_slugs: str,
    fhir_bundle: dict[str, Any] | None = None,
    on_progress: ProgressFn | None = None,
) -> dict:
    result: dict | None = None
    for kind, payload in iter_consult_review_pipeline(
        full_text=full_text,
        n_files=n_files,
        consult_docs_meta=consult_docs_meta,
        pdf_warnings=pdf_warnings,
        content_signature=content_signature,
        category_slugs=category_slugs,
        fhir_bundle=fhir_bundle,
        on_progress=on_progress,
    ):
        if kind == "done":
            result = payload
    if result is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Пустой результат пайплайна consult-review")
    return result


def sse_encode_progress(stage: str, pct: int, label_ru: str, partial: dict | None = None) -> str:
    payload = {
        "type": "progress",
        "stage": stage,
        "pct": pct,
        "label_ru": label_ru,
        "partial": partial or {},
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_encode_done(result: dict) -> str:
    return f"data: {json.dumps({'type': 'done', 'result': result}, ensure_ascii=False)}\n\n"


def sse_encode_error(detail: str, status: int = 500) -> str:
    return f"data: {json.dumps({'type': 'error', 'detail': detail, 'status': status}, ensure_ascii=False)}\n\n"
