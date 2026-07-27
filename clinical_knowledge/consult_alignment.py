"""Детерминированные карточки согласования КЗ с МКБ, КП и НПА."""
from __future__ import annotations

import os
import re
from typing import Any, Callable

from clinical_knowledge.consult_criteria_enrichment import (
    coverage_with_evidence,
    diagnosis_assessment_lines,
    expand_kz_blob,
    enrich_kp_card,
    filter_kp_items_by_demographics,
    finalize_completeness_card,
    kp_coverage_comment,
    kz_source_label,
    maybe_apply_criteria_narrative,
    section_text_for_block,
    verify_protocol_excerpt,
)
from clinical_knowledge.consult_evidence_quality import is_kp_checklist_item
from clinical_knowledge.consult_schema import ConsultationDocument
from clinical_knowledge.dispensary_regulations import (
    completeness_regulation_ref,
    follow_up_mentioned_in_text,
    lookup_follow_up_expectations,
)
from clinical_knowledge.kz_block_sources import (
    ALIGNMENT_CARD_ORDER,
    ALIGNMENT_CARD_TITLES,
    SOURCE_KIND_LABELS,
)
from clinical_knowledge.kz_clinical_context import (
    build_clinical_context,
    format_anamnesis_excerpt,
    format_evaluation_basis,
    protocol_pick_comment,
    rank_kp_items_by_context,
)
from clinical_knowledge.protocol_icd_profile_index import merge_profiles_with_index
from clinical_knowledge.kravira_sop_rules import evaluate_sop_block, merge_sop_into_card
from clinical_knowledge.meaningful_excerpt import excerpt_or_empty, meaningful_excerpt
from clinical_knowledge.semantic_rule_fallback import fuzzy_term_in_text

import icd_mkb

GetChunksFn = Callable[[str], list[dict[str, Any]]]


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _norm_tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[а-яёa-z]{4,}", (s or "").lower()) if len(t) >= 4}


def _title_match_score(diagnosis_text: str, ru_title: str | None) -> float:
    if not ru_title or not diagnosis_text:
        return 0.0
    dt = _norm_tokens(diagnosis_text)
    rt = _norm_tokens(ru_title)
    if not dt or not rt:
        return 0.0
    overlap = dt & rt
    return len(overlap) / max(len(rt), 1)


def _excerpt(text: str | None, limit: int = 280) -> str:
    return meaningful_excerpt(text, limit=limit) or excerpt_or_empty(text, limit=limit)


_MED_SHORT = re.compile(
    r"узи|кт\b|мрт|экг|ривароксабан|апиксабан|варфарин|антибиот|колоноскоп|"
    r"фгдс|спиромет|холтер|коагул|оак\b",
    re.I,
)


def _item_mentioned(kz_blob: str, item: str) -> bool:
    ok, _, _ = fuzzy_term_in_text(kz_blob, item)
    if ok:
        return True
    low = (kz_blob or "").lower()
    for m in _MED_SHORT.finditer(item or ""):
        if m.group(0).lower() in low:
            return True
    head = re.split(r"[—\-–;]", item or "")[0].strip()
    if len(head) >= 10:
        ok2, _, _ = fuzzy_term_in_text(kz_blob, head)
        if ok2:
            return True
    return False


def _coverage_pct(
    required: list[str],
    kz_blob: str,
    *,
    meta: list[dict[str, Any]] | None = None,
) -> tuple[int, list[str], list[str]]:
    if not required:
        return 0, [], []
    from clinical_knowledge.kz_chunk_match import match_kp_item_to_kz

    found: list[str] = []
    missing: list[str] = []
    meta_by_text: dict[str, dict[str, Any]] = {}
    for m in meta or []:
        if isinstance(m, dict):
            t = str(m.get("text") or "")
            meta_by_text[t[:80].lower()] = m

    req_slice = required[:12]
    total_weight = 0.0
    got_weight = 0.0
    for item in req_slice:
        key = item[:80].lower()
        mrow = meta_by_text.get(key, {})
        obligation = str(mrow.get("obligation") or "recommended")
        weight = 1.5 if obligation == "required" else 1.0
        total_weight += weight
        entities = (mrow.get("entities") or {}) if isinstance(mrow.get("entities"), dict) else {}
        km = match_kp_item_to_kz(item, kz_blob, entities=entities)
        mentioned = km["kz_match"] in ("found", "partial") or _item_mentioned(kz_blob, item)
        if mentioned:
            found.append(item)
            got_weight += weight * (1.0 if km["kz_match"] == "found" else 0.85)
        else:
            missing.append(item)

    pct = round(100 * got_weight / total_weight) if total_weight else 0
    return pct, found, missing


def _kz_exam_blob(doc: ConsultationDocument) -> str:
    parts = [
        doc.sections.recommendations_exams or "",
        doc.sections.exam_results or "",
        doc.sections.general_recommendations or "",
    ]
    for ex in doc.performed_exams or []:
        parts.append(getattr(ex, "raw_text", None) or ex.exam_name or "")
    return "\n".join(p for p in parts if p)


def _kz_treatment_blob(doc: ConsultationDocument) -> str:
    parts = [doc.sections.recommendations_treatment or ""]
    for m in doc.medications or []:
        parts.append(m.raw_text or m.drug_name or "")
    return "\n".join(p for p in parts if p)


def _completeness_score(present: bool, *, min_chars: int = 20, text: str = "") -> int:
    if not present:
        return 25
    if len((text or "").strip()) < min_chars:
        return 55
    if "undefined" in (text or "").lower():
        return 45
    return 88


def _card(
    block_id: str,
    *,
    score_pct: int,
    comment_ru: str,
    conclusion_excerpt: str = "",
    protocol_excerpt: str = "",
    protocol_section: str = "",
    protocol_page: str = "",
    source_kind: str = "completeness",
    source_label: str | None = None,
    protocol_path: str = "",
    chunk_id: str | None = None,
    protocol_title: str = "",
    findings_ru: list[str] | None = None,
    gaps_ru: list[str] | None = None,
    context_ru: str = "",
    reference_ru: str = "",
    item_details: list[dict[str, Any]] | None = None,
    gap_protocol_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "name_ru": ALIGNMENT_CARD_TITLES.get(block_id, block_id),
        "score_pct": max(0, min(100, int(score_pct))),
        "comment_ru": comment_ru,
        "conclusion_excerpt": conclusion_excerpt,
        "protocol_excerpt": protocol_excerpt,
        "protocol_section": protocol_section,
        "protocol_page": protocol_page,
        "source_kind": source_kind,
        "source_label": source_label or SOURCE_KIND_LABELS.get(source_kind, source_kind),
        "protocol_path": protocol_path,
        "chunk_id": chunk_id,
        "protocol_title": protocol_title,
        "findings_ru": list(findings_ru or []),
        "gaps_ru": list(gaps_ru or []),
        "context_ru": context_ru,
        "reference_ru": reference_ru,
        "item_details": list(item_details or []),
        "gap_protocol_refs": list(gap_protocol_refs or []),
        "deterministic": True,
    }


def _basename(path: str) -> str:
    p = (path or "").replace("\\", "/").strip()
    return p.split("/")[-1][:80] if p else ""


def _format_kp_cite(cite: dict[str, Any], fallback_lines: list[str]) -> tuple[str, str, str]:
    text = (cite.get("text") or "").strip()
    if not text and fallback_lines:
        text = ""
    verified = verify_protocol_excerpt(text, cite=cite) if text else ""
    if not verified and fallback_lines:
        verified = verify_protocol_excerpt("; ".join(fallback_lines[:2]))
    section = (cite.get("section_title") or "КП").strip()
    page = str(cite.get("page_from") or "")
    path = _basename(cite.get("path") or "")
    header = " · ".join(x for x in [path, f"стр. {page}" if page else ""] if x)
    return verified, section, header


def _mkb_reference_line(code: str, title: str | None) -> str:
    """Строка эталона МКБ без дублирования кода (N72: N72 - …)."""
    c = (code or "").strip()
    t = (title or "").strip()
    if not c:
        return t
    if not t:
        return c
    upper_t = t.upper()
    c_base = c.upper()
    if upper_t.startswith(c_base) or upper_t.startswith(c_base.replace(".", "")):
        return t
    return f"{c} - {t}"


def _diagnosis_card(doc: ConsultationDocument, icd_codes: list[str]) -> dict[str, Any]:
    diag_text = doc.sections.diagnosis_text or ""
    if not diag_text and doc.diagnoses:
        diag_text = "; ".join(d.diagnosis_name or d.raw_text for d in doc.diagnoses if d.raw_text)

    comments, scores, mkb_excerpts = diagnosis_assessment_lines(doc)

    if not doc.diagnoses:
        if icd_codes:
            for code in icd_codes[:3]:
                if icd_mkb.is_code_in_ru_reference(code):
                    title = icd_mkb.ru_title(code)
                    mkb_excerpts.append(_mkb_reference_line(code, title or "—"))
            scores.append(60)
            comments.append("Диагнозы структурно не разобраны; коды извлечены из текста.")
        else:
            scores.append(20)
            comments.append("Диагноз и код МКБ-10 не распознаны.")

    score = round(sum(scores) / len(scores)) if scores else 30
    return _card(
        "diagnosis",
        score_pct=score,
        comment_ru=" · ".join(comments) or "Оценка по справочнику МКБ-10.",
        conclusion_excerpt=section_text_for_block(doc, "diagnosis", ctx=None),
        protocol_excerpt="; ".join(mkb_excerpts)[:400],
        protocol_section="Справочник МКБ-10 (RU)",
        source_kind="mkb",
    )


def _kp_title(protocol_matches: list[dict[str, Any]] | None, profile: dict[str, Any]) -> str:
    from clinical_knowledge.protocol_links import protocol_display_name

    for m in protocol_matches or []:
        title = protocol_display_name(
            str(m.get("source_path") or m.get("path") or ""),
            registry_title=str(m.get("title") or ""),
        )
        if title and title != "Протокол":
            return title[:120]
    paths = profile.get("paths") or []
    if paths:
        return protocol_display_name(str(paths[0]))[:120]
    return ""


def _complaints_card(doc: ConsultationDocument, ctx: dict[str, Any]) -> dict[str, Any]:
    text = (doc.sections.complaints or "").strip()
    present = bool(text)
    score = _completeness_score(present, text=text, min_chars=12)
    if not present:
        comment = "Жалобы не описаны - по СОП Кравira раздел обязателен."
    elif len(text) < 25:
        comment = "Жалобы указаны кратко; по СОП нужна детализация (давность, характер, динамика)."
        score = min(score, 60)
    else:
        comment = "Жалобы заполнены; проверьте детализацию по СОП (характер, давность, динамика)."
    if doc.extraction_quality.has_undefined and "undefined" in text.lower():
        score = min(score, 45)
        comment += " В тексте есть незаполненные поля."
    return _card(
        "complaints",
        score_pct=score,
        comment_ru=comment,
        conclusion_excerpt=section_text_for_block(doc, "complaints", ctx),
        source_kind="completeness",
    )


def _anamnesis_card(doc: ConsultationDocument, ctx: dict[str, Any]) -> dict[str, Any]:
    disease = (ctx.get("anamnesis_disease") or "").strip()
    life = (ctx.get("anamnesis_life") or "").strip()
    present = bool(disease or life)
    score = _completeness_score(present, text=disease or life, min_chars=20)
    parts: list[str] = []
    if disease:
        parts.append("Анамнез заболевания заполнен.")
    else:
        parts.append("Анамнез заболевания не описан.")
        score = min(score, 50)
    if life:
        parts.append("Анамнез жизни заполнен.")
    else:
        parts.append("Анамнез жизни не указан.")
        if score > 72:
            score -= 10
    return _card(
        "anamnesis",
        score_pct=score,
        comment_ru=" ".join(parts),
        conclusion_excerpt=section_text_for_block(doc, "anamnesis", ctx),
        source_kind="completeness",
    )


def _completeness_section_card(
    block_id: str,
    doc: ConsultationDocument,
    *,
    text: str,
    present: bool,
) -> dict[str, Any]:
    score = _completeness_score(present, text=text)
    if not present or score < 50:
        comment = "Объективный статус не описан или слишком краткий - по СОП обязательны осмотр и витальные параметры."
    elif len((text or "").strip()) < 30:
        comment = "Объективный статус краткий; добавьте витальные параметры и локальный статус."
    else:
        comment = "Объективный статус заполнен; по СОП проверьте АД, пульс, температуру и локальный статус."
    if doc.extraction_quality.has_undefined and block_id == "objective_status":
        comment += " Есть незаполненные поля."
        score = min(score, 50)
    return _card(
        block_id,
        score_pct=score,
        comment_ru=comment,
        conclusion_excerpt=section_text_for_block(doc, block_id, None),
        source_kind="completeness",
    )


def _structured_items(
    protocol_paths: list[str] | None,
    ctx: dict[str, Any],
    kind: str,
) -> list[tuple[str, str]]:
    """Чистые пункты обследований/лечения из структурных полей ProtocolSummary (Э3).

    Возвращает [(text, obligation)]. Заменяет обрывки прозы из чанков
    (`profile["diagnostics"]`) на реальные названия обследований/препаратов -
    тогда каталог синонимов (Э2) начинает работать. Пусто → вызывающий код
    откатывается на прежний путь по чанкам.
    """
    if not _env_bool("CONSULT_STRUCTURED_ITEMS", True):
        return []
    try:
        from clinical_knowledge.protocol_summary.loader import load_summary_by_path
    except Exception:
        return []

    icd = {str(c).upper() for c in (ctx.get("icd_codes") or []) if c}
    icd_roots = {c[:3] for c in icd}
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(text: str | None, obligation: str) -> None:
        t = (text or "").strip()
        if not t or len(t) < 5 or len(t) > 180:
            return
        key = t.lower()
        if key in seen or not is_kp_checklist_item(t):
            return
        seen.add(key)
        out.append((t, obligation))

    for path in (protocol_paths or [])[:4]:
        try:
            summary = load_summary_by_path(str(path), usable_only=False)
        except Exception:
            summary = None
        if summary is None:
            continue
        conds = list(summary.conditions or [])
        if icd:
            picked = [
                c for c in conds
                if any(
                    (str(code).upper() in icd) or (str(code)[:3].upper() in icd_roots)
                    for code in (c.icd10_codes or [])
                )
            ]
            conds = picked or conds
        for c in conds:
            if kind == "exams":
                for ex in (c.required_exams or []):
                    ob = "required" if str(getattr(ex, "requirement_level", "")) == "required" else "recommended"
                    _add(getattr(ex, "name", None), ob)
                for ex in (c.conditional_exams or []):
                    _add(getattr(ex, "name", None), "recommended")
            else:  # treatment
                tr = getattr(c, "treatment", None)
                if not tr:
                    continue
                for g in (getattr(tr, "drug_groups", None) or []):
                    _add(getattr(g, "drug_group", None), "recommended")
                for d in (getattr(tr, "drugs", None) or []):
                    _add(getattr(d, "drug_name", None) or getattr(d, "active_substance", None)
                         or getattr(d, "drug_group", None), "recommended")
                for pr in (getattr(tr, "procedures", None) or []):
                    _add(getattr(pr, "name", None), "recommended")
                for sg in (getattr(tr, "surgery", None) or []):
                    _add(getattr(sg, "name", None), "recommended")
                for nd in (getattr(tr, "non_drug", None) or []):
                    _add(getattr(nd, "text", None), "recommended")
        if len(out) >= 14:
            break
    return out[:14]


def _exams_card(
    doc: ConsultationDocument,
    profile: dict[str, Any],
    ctx: dict[str, Any],
    protocol_matches: list[dict[str, Any]] | None,
    *,
    get_chunks: GetChunksFn | None = None,
    protocol_paths: list[str] | None = None,
) -> dict[str, Any]:
    diag_pool = filter_kp_items_by_demographics(
        list(profile.get("diagnostics") or []),
        list(profile.get("diagnostics_meta") or []),
        ctx,
    )
    ranked = rank_kp_items_by_context(
        diag_pool,
        ctx,
        meta=list(profile.get("diagnostics_meta") or []),
        limit=12,
    )
    required = [r["text"] for r in ranked if is_kp_checklist_item(r.get("text") or "")]
    struct_exams = _structured_items(protocol_paths or profile.get("paths"), ctx, "exams")
    if struct_exams:
        ranked = [{"text": t, "obligation": ob} for t, ob in struct_exams]
        required = [t for t, _ in struct_exams]
    kz_blob = expand_kz_blob(doc, "exams")
    raw_text = (getattr(doc, "raw_text", None) or "").strip()
    cite = next(
        (c for c in (profile.get("cites") or []) if c.get("chunk_type") in ("diagnostics", "criteria_block", "table")),
        (profile.get("cites") or [{}])[0] if profile.get("cites") else {},
    )
    if not cite.get("chunk_id") and profile.get("paths"):
        try:
            from clinical_knowledge.kz_chunk_match import best_chunk_for_items

            all_chunks: list[dict[str, Any]] = []
            paths_for_chunks = (protocol_paths or profile.get("paths") or [])[:3]
            for pth in paths_for_chunks:
                if get_chunks:
                    from clinical_knowledge.consult_memory import cap_chunks_for_consult

                    all_chunks.extend(cap_chunks_for_consult(get_chunks(pth) or []))
            best = best_chunk_for_items(
                all_chunks,
                chunk_types=("diagnostics", "criteria_block", "table"),
                icd_codes=list(ctx.get("icd_codes") or []),
            )
            if best:
                cite = {
                    "path": best.get("source_path"),
                    "chunk_id": best.get("chunk_id"),
                    "chunk_type": best.get("chunk_type"),
                    "page_from": best.get("page_from"),
                    "text": best.get("text"),
                    "section_title": best.get("section_title"),
                }
        except Exception:
            pass
    proto_text, proto_section, proto_header = _format_kp_cite(cite, required)
    kp_title = _kp_title(protocol_matches, profile)
    pick_note = protocol_pick_comment(ctx, protocol_matches)
    basis = format_evaluation_basis(ctx, protocol_matches)

    if pick_note and not required:
        return _card(
            "exams",
            score_pct=40 if kz_blob else 25,
            comment_ru=pick_note,
            conclusion_excerpt=section_text_for_block(doc, "exams", ctx),
            protocol_excerpt=proto_text,
            protocol_section=proto_section,
            protocol_page=str(cite.get("page_from") or ""),
            source_kind="kp",
            protocol_path=cite.get("path") or "",
            chunk_id=cite.get("chunk_id"),
            context_ru=basis,
        )

    if not required:
        comment = f"КП «{kp_title}»: обследования для сопоставления не извлечены." if kp_title else pick_note or "Обследования КП не извлечены."
        return _card(
            "exams",
            score_pct=55 if kz_blob else 35,
            comment_ru=comment,
            conclusion_excerpt=section_text_for_block(doc, "exams", ctx),
            protocol_excerpt=proto_text,
            protocol_section=proto_section,
            protocol_page=str(cite.get("page_from") or ""),
            source_kind="kp",
            protocol_path=cite.get("path") or "",
            chunk_id=cite.get("chunk_id"),
            context_ru=basis,
        )

    pct, found, missing, details = coverage_with_evidence(
        required, kz_blob, meta=ranked, raw_text=raw_text,
    )
    comment = kp_coverage_comment(
        kp_title, found, missing, details, kind="обследований", pick_note=pick_note,
    )
    if not kz_blob:
        pct = min(pct, 35)
        comment += " Назначения и результаты обследований в КЗ не распознаны."

    card = _card(
        "exams",
        score_pct=max(pct, 15) if kz_blob else min(pct, 40),
        comment_ru=comment,
        conclusion_excerpt=section_text_for_block(doc, "exams", ctx),
        protocol_excerpt=(proto_header + ": " if proto_header else "") + proto_text,
        protocol_section=proto_section or "Обследование",
        protocol_page=str(cite.get("page_from") or ""),
        source_kind="kp",
        protocol_path=cite.get("path") or "",
        chunk_id=cite.get("chunk_id"),
        context_ru=basis,
    )
    card["_cites"] = list(profile.get("cites") or [])
    enrich_kp_card(
        card,
        details=details,
        kz_blob=kz_blob,
        cite=cite,
        ranked=ranked,
        get_chunks=get_chunks,
        protocol_paths=protocol_paths or profile.get("paths"),
    )
    card.pop("_cites", None)
    return card


def _treatment_card(
    doc: ConsultationDocument,
    profile: dict[str, Any],
    ctx: dict[str, Any],
    protocol_matches: list[dict[str, Any]] | None,
    *,
    get_chunks: GetChunksFn | None = None,
    protocol_paths: list[str] | None = None,
) -> dict[str, Any]:
    pool_raw = list(profile.get("medications") or []) + list(profile.get("treatment") or [])
    pool = filter_kp_items_by_demographics(
        pool_raw,
        list(profile.get("medications_meta") or []) + list(profile.get("treatment_meta") or []),
        ctx,
    )
    ranked = rank_kp_items_by_context(
        pool,
        ctx,
        meta=list(profile.get("medications_meta") or []),
        limit=12,
    )
    required = [r["text"] for r in ranked if is_kp_checklist_item(r.get("text") or "")]
    struct_treat = _structured_items(protocol_paths or profile.get("paths"), ctx, "treatment")
    if struct_treat:
        ranked = [{"text": t, "obligation": ob} for t, ob in struct_treat]
        required = [t for t, _ in struct_treat]
    kz_blob = expand_kz_blob(doc, "treatment")
    raw_text = (getattr(doc, "raw_text", None) or "").strip()
    cite = next(
        (c for c in (profile.get("cites") or []) if c.get("chunk_type") in ("pharmacotherapy", "treatment", "drug_list")),
        (profile.get("cites") or [{}])[0] if profile.get("cites") else {},
    )
    proto_text, proto_section, proto_header = _format_kp_cite(cite, required)
    kp_title = _kp_title(protocol_matches, profile)
    pick_note = protocol_pick_comment(ctx, protocol_matches)
    basis = format_evaluation_basis(ctx, protocol_matches)

    if pick_note and not required:
        return _card(
            "treatment",
            score_pct=45 if kz_blob else 25,
            comment_ru=pick_note,
            conclusion_excerpt=section_text_for_block(doc, "exams", ctx),
            protocol_excerpt=proto_text,
            protocol_section=proto_section,
            source_kind="kp",
            protocol_path=cite.get("path") or "",
            chunk_id=cite.get("chunk_id"),
            context_ru=basis,
        )

    if not required:
        comment = f"КП «{kp_title}»: рекомендации по лечению не извлечены." if kp_title else pick_note or "Лечение по КП не извлечено."
        return _card(
            "treatment",
            score_pct=60 if kz_blob else 35,
            comment_ru=comment,
            conclusion_excerpt=section_text_for_block(doc, "exams", ctx),
            protocol_excerpt=proto_text,
            protocol_section=proto_section,
            protocol_page=str(cite.get("page_from") or ""),
            source_kind="kp",
            protocol_path=cite.get("path") or "",
            chunk_id=cite.get("chunk_id"),
            context_ru=basis,
        )

    pct, found, missing, details = coverage_with_evidence(
        required, kz_blob, meta=ranked, raw_text=raw_text,
    )
    comment = kp_coverage_comment(
        kp_title, found, missing, details, kind="назначений по лечению", pick_note=pick_note,
    )
    if not kz_blob:
        pct = min(pct, 35)
        comment += " Назначения в КЗ не распознаны."

    card = _card(
        "treatment",
        score_pct=max(pct, 20) if kz_blob else min(pct, 45),
        comment_ru=comment,
        conclusion_excerpt=section_text_for_block(doc, "treatment", ctx),
        protocol_excerpt=(proto_header + ": " if proto_header else "") + proto_text,
        protocol_section=proto_section or "Лечение",
        protocol_page=str(cite.get("page_from") or ""),
        source_kind="kp",
        protocol_path=cite.get("path") or "",
        chunk_id=cite.get("chunk_id"),
        context_ru=basis,
    )
    card["_cites"] = list(profile.get("cites") or [])
    enrich_kp_card(
        card,
        details=details,
        kz_blob=kz_blob,
        cite=cite,
        ranked=ranked,
        get_chunks=get_chunks,
        protocol_paths=protocol_paths or profile.get("paths"),
    )
    card.pop("_cites", None)
    return card


def _follow_up_card(
    doc: ConsultationDocument,
    icd_codes: list[str],
    profile: dict[str, Any],
) -> dict[str, Any]:
    blob = expand_kz_blob(doc, "follow_up")

    reg = lookup_follow_up_expectations(icd_codes)
    kp_mon = list(profile.get("monitoring") or [])
    has_follow = follow_up_mentioned_in_text(blob, min_months=reg.get("min_interval_months"))

    hints = list(reg.get("follow_up_hints") or [])
    if kp_mon:
        hints.append(kp_mon[0][:200])

    proto_excerpt = ""
    if kp_mon:
        proto_excerpt = verify_protocol_excerpt(kp_mon[0][:400])
        source_kind = "kp"
        section = "Диспансерное наблюдение (КП)"
    else:
        raw_reg = (reg.get("conclusion_requirement") or hints[0] if hints else "")[:280]
        proto_excerpt = verify_protocol_excerpt(raw_reg) or raw_reg[:280]
        source_kind = "regulation"
        section = reg.get("regulation_source") or "НПА № 127"

    if has_follow:
        score = 92
        comment = "В КЗ указаны сроки/рекомендации контрольного наблюдения."
    elif doc.sections.recommendations_treatment or doc.medications:
        score = 50
        comment = "Назначено лечение, но сроки контрольного наблюдения не описаны явно."
    else:
        score = 65
        comment = "Контрольное наблюдение не требуется или не описано."

    if hints and not has_follow:
        comment += f" По НПА/КП: {hints[0][:120]}."

    return _card(
        "follow_up",
        score_pct=score,
        comment_ru=comment,
        conclusion_excerpt=section_text_for_block(doc, "follow_up", None),
        protocol_excerpt=proto_excerpt,
        protocol_section=section,
        source_kind=source_kind,
    )


def _limitations_card(
    profile: dict[str, Any],
    ctx: dict[str, Any],
    protocol_matches: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    parts: list[str] = []
    missing = list(ctx.get("missing_for_protocol_pick") or [])
    if missing:
        parts.append(f"Для подбора КП не указаны: {', '.join(missing)}.")
    pick_note = protocol_pick_comment(ctx, protocol_matches)
    if pick_note:
        parts.append(pick_note)
    elif protocol_matches:
        title = _kp_title(protocol_matches, profile)
        sc = float((protocol_matches[0] or {}).get("match_score") or 0)
        if title:
            parts.append(f"Подобран КП «{title}» (соответствие {sc:.0f}%).")
    elif not (ctx.get("icd_codes") or []):
        parts.append("Код МКБ-10 не указан — сравнение с КП невозможно.")
    elif not profile.get("paths"):
        parts.append("Клинический протокол по данным КЗ не найден.")
    if not parts:
        parts.append("Достаточно данных для сопоставления с КП.")
    return _card(
        "limitations",
        score_pct=100,
        comment_ru=" ".join(parts),
        source_kind="limitations",
        source_label="Ограничения",
    )


_COMPLETENESS_NPA_BLOCKS = ("complaints", "anamnesis", "objective_status")


def _attach_npa_to_completeness(card: dict[str, Any]) -> None:
    """G: показать НПА-эталон (127) в карточке полноты, СОП оставить деталью."""
    bid = str(card.get("block_id") or "")
    if bid not in _COMPLETENESS_NPA_BLOCKS:
        return
    ref = completeness_regulation_ref(bid)
    if not ref:
        return
    npa_src = ref.get("regulation_source") or "Постановление № 127"
    npa_excerpt = ref.get("excerpt_ru") or ""
    sop_ref = (card.get("reference_ru") or "").strip()
    combined = f"НПА - {npa_src}: {npa_excerpt}."
    if sop_ref and "127" not in sop_ref:
        combined += f" Внутр. стандарт - {sop_ref}"
    card["reference_ru"] = combined
    card["protocol_excerpt"] = npa_excerpt or card.get("protocol_excerpt") or ""
    card["protocol_section"] = f"{npa_src} · СОП № 2 Кравира"
    card["regulation_source_ru"] = npa_src
    card["regulation_url"] = ref.get("url") or ""
    card["source_label"] = "НПА № 127 / СОП"


def _kp_confidence(
    ctx: dict[str, Any],
    protocol_matches: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Оценка уверенности подбора КП для E-quick."""
    matches = protocol_matches or []
    top = matches[0] if matches else {}
    try:
        score = float(top.get("match_score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    blob = f"{top.get('title') or ''} {top.get('path') or ''}".lower()
    stationary = ("стационар" in blob) and ("амбулатор" not in blob)
    scope_mismatch = stationary and (ctx.get("setting") or "ambulatory") == "ambulatory"
    low = bool(score) and score < 45.0
    return {"score": score, "scope_mismatch": scope_mismatch, "low": low}


def _apply_kp_confidence_guard(card: dict[str, Any], conf: dict[str, Any]) -> None:
    """E-quick: при слабом подборе КП не штрафовать жёстко за «пробелы»."""
    if str(card.get("block_id") or "") not in ("exams", "treatment"):
        return
    if not (conf.get("low") or conf.get("scope_mismatch")):
        return
    floor = 50
    if int(card.get("score_pct") or 0) < floor:
        card["score_pct"] = floor
    reasons: list[str] = []
    if conf.get("scope_mismatch"):
        reasons.append("подобран протокол для стационара, а КЗ амбулаторное")
    if conf.get("low"):
        reasons.append(f"низкое соответствие КП ({conf.get('score', 0):.0f}%)")
    note = (
        "Подбор КП низкой уверенности (" + "; ".join(reasons)
        + ") - пробелы информативны, но не штрафуются жёстко."
    )
    card["kp_low_confidence"] = True
    card["confidence_note_ru"] = note
    base_comment = (card.get("comment_ru") or "").rstrip()
    if note not in base_comment:
        card["comment_ru"] = (base_comment + " " + note).strip()


def build_consult_alignment(
    doc: ConsultationDocument,
    *,
    protocol_paths: list[str],
    icd_codes: list[str],
    get_chunks: GetChunksFn,
    query: str = "",
    protocol_matches: list[dict[str, Any]] | None = None,
    specialty_slug: str | None = None,
    specialty_label: str | None = None,
) -> dict[str, Any]:
    """Построить детерминированные карточки и criteria для UI."""
    ctx = build_clinical_context(
        doc,
        icd_codes,
        specialty_slug=specialty_slug,
        specialty_label=specialty_label,
    )
    clinical_query = (query or "").strip() or str(ctx.get("clinical_query") or "")
    profile = merge_profiles_with_index(
        protocol_paths, icd_codes, get_chunks, query=clinical_query
    )
    paths_for_cards = list((profile.get("paths") or protocol_paths or [])[:4])
    cards: list[dict[str, Any]] = []

    cards.append(_diagnosis_card(doc, icd_codes))
    cards.append(_complaints_card(doc, ctx))
    cards.append(_anamnesis_card(doc, ctx))
    cards.append(
        _completeness_section_card(
            "objective_status",
            doc,
            text=doc.sections.objective_status or "",
            present=bool(doc.sections.objective_status),
        )
    )
    cards.append(_exams_card(doc, profile, ctx, protocol_matches, get_chunks=get_chunks, protocol_paths=paths_for_cards))
    cards.append(_treatment_card(doc, profile, ctx, protocol_matches, get_chunks=get_chunks, protocol_paths=paths_for_cards))
    cards.append(_follow_up_card(doc, icd_codes, profile))
    cards.append(_limitations_card(profile, ctx, protocol_matches))

    kp_conf = _kp_confidence(ctx, protocol_matches)
    for card in cards:
        bid = str(card.get("block_id") or "")
        if bid and bid != "limitations":
            merge_sop_into_card(card, evaluate_sop_block(doc, bid))
            finalize_completeness_card(card)
        # A + G: НПА-эталон полноты (127) поверх СОП для карточек осмотра.
        _attach_npa_to_completeness(card)
        # E-quick: гасим ложные пробелы КП при слабом/нерелевантном подборе.
        _apply_kp_confidence_guard(card, kp_conf)

    by_id = {c["block_id"]: c for c in cards}
    ordered = [by_id[bid] for bid in ALIGNMENT_CARD_ORDER if bid in by_id]

    scorable = [c for c in ordered if c["block_id"] != "limitations"]
    mean_score = round(sum(c["score_pct"] for c in scorable) / len(scorable)) if scorable else 0

    limitations = " ".join(
        c["comment_ru"] for c in ordered if c["block_id"] == "limitations"
    )

    criteria = [_card_to_criterion(c) for c in ordered if c["block_id"] != "limitations"]
    kz_file = (doc.source_file or "").strip()
    kz_label = kz_source_label(kz_file) if kz_file else ""
    criteria = maybe_apply_criteria_narrative(criteria)

    return {
        "alignment_cards": ordered,
        "criteria": criteria,
        "kz_source_file": kz_file,
        "kz_source_label": kz_label,
        "alignment_mean_score": mean_score,
        "limitations_ru": limitations,
        "audit_trail": {
            "protocol_matches": (protocol_matches or [])[:8],
            "icd_codes": list(icd_codes or [])[:8],
            "protocol_paths": list(protocol_paths or [])[:8],
            "profile_diagnostics": len(profile.get("diagnostics") or []),
            "profile_treatment": len(profile.get("treatment") or []) + len(profile.get("medications") or []),
        },
        "protocol_profile": {
            "paths": profile.get("paths") or [],
            "diagnostics_count": len(profile.get("diagnostics") or []),
            "medications_count": len(profile.get("medications") or []),
        },
    }


def _card_to_criterion(card: dict[str, Any]) -> dict[str, Any]:
    out = {k: card.get(k) for k in (
        "name_ru", "score_pct", "comment_ru", "conclusion_excerpt",
        "protocol_excerpt", "protocol_section", "protocol_page",
        "source_kind", "source_label", "protocol_path", "chunk_id", "deterministic",
        "findings_ru", "gaps_ru", "context_ru", "reference_ru", "block_id",
        "item_details", "gap_protocol_refs", "comment_narrative_llm",
        "kz_source_file", "kz_source_label", "protocol_title",
    )}
    return out


def merge_alignment_into_review(review: dict[str, Any], alignment: dict[str, Any]) -> None:
    """Подменить LLM-критерии детерминированными (in-place)."""
    if not _env_bool("CONSULT_ALIGNMENT_PRIMARY", True):
        return
    criteria = alignment.get("criteria") or []
    if criteria:
        review["criteria"] = criteria
        review["criteria_source"] = "deterministic_alignment"
    if alignment.get("limitations_ru") and not (review.get("limitations_ru") or "").strip():
        review["limitations_ru"] = alignment["limitations_ru"]
    if alignment.get("kz_source_file"):
        review["kz_source_file"] = alignment["kz_source_file"]
    if alignment.get("kz_source_label"):
        review["kz_source_label"] = alignment["kz_source_label"]
    review["alignment_cards"] = alignment.get("alignment_cards") or []
    review["alignment_mean_score"] = alignment.get("alignment_mean_score")


BLOCK_TO_SCORE_KEY: dict[str, str] = {
    "diagnosis": "diagnosis_score",
    "exams": "required_exams_score",
    "treatment": "treatment_score",
    "follow_up": "follow_up_score",
}


def sync_structured_with_alignment(
    structured_analysis: dict[str, Any] | None,
    alignment: dict[str, Any] | None,
) -> None:
    """Связать 8 блоков structured с alignment_cards (in-place)."""
    if not structured_analysis or not alignment:
        return
    comp = structured_analysis.get("compliance")
    if not isinstance(comp, dict):
        return
    sb = comp.get("score_breakdown") or {}
    by_block: dict[str, Any] = {}
    for card in alignment.get("alignment_cards") or []:
        bid = str(card.get("block_id") or "")
        sk = BLOCK_TO_SCORE_KEY.get(bid)
        by_block[bid] = {
            "name_ru": card.get("name_ru"),
            "alignment_score": card.get("score_pct"),
            "structured_score": sb.get(sk) if sk else None,
            "source_kind": card.get("source_kind"),
            "source_label": card.get("source_label"),
        }
    comp["alignment_by_block"] = by_block
    audit = alignment.get("audit_trail")
    if audit:
        structured_analysis["audit_trail"] = audit
    structured_analysis["compliance"] = comp


def alignment_to_evidence_items(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """EvidenceMapItem-подобные записи из alignment для structured отчёта."""
    out: list[dict[str, Any]] = []
    for card in cards:
        if card.get("block_id") in ("limitations",):
            continue
        out.append({
            "block_id": card.get("block_id"),
            "rule_title_ru": card.get("name_ru"),
            "decision_ru": f"{card.get('score_pct')}%",
            "source_kind": card.get("source_kind"),
            "protocol_excerpt": (card.get("protocol_excerpt") or "")[:400],
            "consultation_excerpt": (card.get("conclusion_excerpt") or "")[:400],
            "protocol_section": card.get("protocol_section"),
            "protocol_page": card.get("protocol_page"),
            "rule_source": "alignment",
        })
    return out


def append_alignment_evidence(
    structured_analysis: dict[str, Any] | None,
    alignment: dict[str, Any] | None,
) -> None:
    if not structured_analysis or not alignment:
        return
    comp = structured_analysis.get("compliance")
    if not isinstance(comp, dict):
        return
    existing = list(comp.get("evidence_map") or [])
    extra = alignment_to_evidence_items(alignment.get("alignment_cards") or [])
    comp["evidence_map"] = existing + extra
    structured_analysis["compliance"] = comp
