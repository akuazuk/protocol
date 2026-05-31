"""Пайплайн проверки КЗ с поэтапным прогрессом (SSE)."""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

ProgressFn = Any  # Callable[[str, int, str, dict], None] — optional legacy


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


def iter_consult_review_pipeline(
    *,
    full_text: str,
    n_files: int,
    consult_docs_meta: list[dict],
    pdf_warnings: list[str],
    content_signature: str,
    category_slugs: str,
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
        yield ("done", cached)
        return

    yield emit("focus", 15, "Анализ текста заключения (фокус запроса)…")
    model = rs.get_gemini()
    synthetic, retrieval_focus_meta = rs._build_consult_review_pipeline_query(model, full_text)

    yield emit("icd", 28, "Подбор кодов МКБ-10 и клинического контекста…")
    icd_analysis, q, q_rag, _, icd_err = rs._infer_icd_pipeline_from_full_query(synthetic, model)
    q_slice_fb = int(__import__("os").environ.get("CONSULT_REVIEW_RAG_QUERY_CHARS", "9000"))
    fallback_synthetic_legacy = (
        "=== Жалобы и вопрос ===\n\n" + full_text[: min(len(full_text), q_slice_fb)]
    )
    if icd_err or not (q_rag or "").strip():
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
    icd_codes_for_lex = merged_icd or (icd_analysis.get("codes_for_retrieval") or None)
    diag_codes_list = (
        icd_merge_meta.get("diag_block_icd_codes")
        if isinstance(icd_merge_meta.get("diag_block_icd_codes"), list)
        else []
    )

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
    boost_merged = list(dict.fromkeys((query_specialties or []) + user_slugs))

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
            "icd": rs._icd_client_payload(icd_analysis),
            "demographics_meta": demographics_meta,
            "consult_retrieval": {
                "icd_codes_lex_merged": merged_icd,
                "diag_block_icd_codes": diag_codes_list,
            },
        },
    )

    yield emit("rules", 45, "Проверка по правилам протоколов…")
    clinical_rules = rs._consult_clinical_rules_pipeline(
        full_text,
        demographics_meta if isinstance(demographics_meta, dict) else {},
        list(merged_icd or []),
        user_slugs,
    )
    matched_path_boost: list[str] = []
    if isinstance(clinical_rules, dict):
        for mp in clinical_rules.get("matched_protocols") or []:
            sp = (mp or {}).get("source_path")
            if sp and sp not in matched_path_boost:
                matched_path_boost.append(sp)
        yield emit("rules_done", 52, "Правила протоколов применены", {"clinical_rules": clinical_rules})

    max_chunks_r = int(__import__("os").environ.get("CONSULT_REVIEW_MAX_CHUNKS", "14"))
    max_per_path_r = int(__import__("os").environ.get("CONSULT_REVIEW_MAX_PER_PATH", "3"))

    yield emit("retrieve", 58, "Поиск фрагментов клинических протоколов…")
    retrieved = rs.retrieve(
        q_rag,
        routing_query=rq,
        category_boost=boost_merged or None,
        user_category_slugs=user_slugs or None,
        icd_codes_for_lex=icd_codes_for_lex,
        path_boost=matched_path_boost or None,
        max_chunks=max_chunks_r,
        max_per_path=max_per_path_r,
    )
    if not retrieved:
        fallback_q = full_text[: min(5500, len(full_text))]
        retrieved = rs.retrieve(
            fallback_q,
            routing_query=full_text[: min(9500, len(full_text))],
            category_boost=boost_merged or None,
            user_category_slugs=user_slugs or None,
            icd_codes_for_lex=(merged_icd or None),
            path_boost=matched_path_boost or None,
            max_chunks=max(14, max_chunks_r + 2),
            max_per_path=max_per_path_r,
        )
    if not retrieved:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail="Не удалось подобрать фрагменты протоколов по тексту PDF — попробуйте другой файл или явно опишите диагноз/МКБ в документе.",
        )

    second_pass_on = rs._consult_rag_second_pass_enabled()
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
                    user_category_slugs=user_slugs or None,
                    icd_codes_for_lex=icd_codes_for_lex,
                    path_boost=matched_path_boost or None,
                    max_chunks=max_chunks_r + bump,
                    max_per_path=max_per_path_r,
                )
                if r2:
                    merge_max = max_chunks_r + bump
                    retrieved = rs._merge_chunk_retrieval_lists(
                        [retrieved, r2],
                        max_chunks=merge_max,
                        max_per_path=max_per_path_r,
                    )
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
    icd_frag_needles = rs._consult_needles_icd_fragments_consult_review(list(diag_codes_list), merged_icd)
    retrieved = rs._consult_sort_retrieval_by_icd_fragments_first(retrieved, icd_frag_needles)
    precise_links, precise_note_ru = rs._consult_precise_links_for_icd_in_fragments(
        retrieved,
        diag_block_icd=list(diag_codes_list),
        merged_icd=merged_icd,
    )

    proto_max = int(__import__("os").environ.get("CONSULT_REVIEW_PROTOCOL_CTX_CHARS", "16500"))
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
            "retrieval_paths": paths_used,
            "consult_protocol_fragments": ui_frags,
            "consult_oncology_flags": oncology,
            "consult_icd_precise_links": precise_links,
            "consult_icd_precise_note_ru": precise_note_ru,
            "consult_retrieval": {
                "focus": retrieval_focus_meta,
                "second_pass": second_pass_diag,
            },
        },
    )

    consult_max = int(__import__("os").environ.get("CONSULT_REVIEW_CONSULT_CHARS", "20000"))
    multi_intro = (
        ""
        if n_files <= 1
        else (
            "Несколько документов: блоки ниже — в порядке загрузки; при оценке учитывай "
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

    yield emit("synthesize", 85, "Формирование оценки и критериев (модель)…")
    review = rs._consult_review_synthesize(
        model,
        consult_excerpt,
        protocol_ctx,
        paths_hint_for_llm,
        extra_context=oncology_extra,
        clinical_rules_context=rules_ctx,
    )

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
        },
        "icd": rs._icd_client_payload(icd_analysis),
        "demographics_meta": demographics_meta,
    }
    if clinical_rules is not None:
        result["clinical_rules"] = clinical_rules
    result["cached_result"] = False
    rs._consult_cache_put(cache_key, result)
    yield emit("done", 100, "Готово", {"review": review})
    yield ("done", result)


def run_consult_review_pipeline(
    *,
    full_text: str,
    n_files: int,
    consult_docs_meta: list[dict],
    pdf_warnings: list[str],
    content_signature: str,
    category_slugs: str,
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
