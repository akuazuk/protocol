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

    # Специальность врача из шапки КЗ — авторитетный якорь рубрики.
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
    # Рубрика врача КЗ — впереди (приоритетный якорь), затем угаданные/выбранные.
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

    max_chunks_r = int(__import__("os").environ.get("CONSULT_REVIEW_MAX_CHUNKS", "12"))
    max_per_path_r = int(__import__("os").environ.get("CONSULT_REVIEW_MAX_PER_PATH", "3"))
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
            "retrieval_paths_count": len(paths_used),
            "protocol_paths_used": paths_used[:8],
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
                with_markdown=True,
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
    }
    if clinical_rules is not None:
        result["clinical_rules"] = clinical_rules
    if structured_analysis is not None:
        result["structured_analysis"] = structured_analysis
    if report_markdown:
        result["report_markdown"] = report_markdown
    if report_html:
        result["report_html"] = report_html

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
    rs._consult_cache_put(cache_key, result)
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
