"""Детерминированные карточки согласования КЗ с МКБ, КП и НПА."""
from __future__ import annotations

import os
import re
from typing import Any, Callable

from clinical_knowledge.consult_schema import ConsultationDocument
from clinical_knowledge.dispensary_regulations import (
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
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t[:limit] if t else ""


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
    findings_ru: list[str] | None = None,
    gaps_ru: list[str] | None = None,
    context_ru: str = "",
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
        "findings_ru": list(findings_ru or []),
        "gaps_ru": list(gaps_ru or []),
        "context_ru": context_ru,
        "deterministic": True,
    }


def _basename(path: str) -> str:
    p = (path or "").replace("\\", "/").strip()
    return p.split("/")[-1][:80] if p else ""


def _format_kp_cite(cite: dict[str, Any], fallback_lines: list[str]) -> tuple[str, str, str]:
    text = (cite.get("text") or "").strip()
    if not text and fallback_lines:
        text = "; ".join(fallback_lines[:3])
    section = (cite.get("section_title") or "КП").strip()
    page = str(cite.get("page_from") or "")
    path = _basename(cite.get("path") or "")
    header = " · ".join(x for x in [path, f"стр. {page}" if page else ""] if x)
    return text[:360], section, header


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

    comments: list[str] = []
    scores: list[int] = []
    mkb_excerpts: list[str] = []

    for d in doc.diagnoses:
        code = d.icd10_code
        if not code:
            comments.append("Код МКБ-10 не указан для одного из диагнозов.")
            scores.append(40)
            continue
        valid = icd_mkb.is_code_in_ru_reference(code)
        title = icd_mkb.ru_title(code)
        desc = icd_mkb.describe_code(code)
        if valid and title:
            mkb_excerpts.append(_mkb_reference_line(code, title))
            match = _title_match_score(d.diagnosis_name or d.raw_text or diag_text, title)
            if match >= 0.35:
                scores.append(95)
                comments.append(f"Код {code} в справочнике МКБ; формулировка согласуется с рубрикой «{title[:80]}».")
            else:
                scores.append(78)
                comments.append(
                    f"Код {code} валиден ({title[:60]}), но текст диагноза слабо совпадает с рубрикой МКБ."
                )
        elif valid:
            scores.append(72)
            comments.append(f"Код {code} найден в справочнике МКБ.")
            if desc.get("title_ru"):
                mkb_excerpts.append(_mkb_reference_line(code, str(desc["title_ru"])))
        else:
            scores.append(30)
            comments.append(f"Код {code} не найден в русском справочнике МКБ-10.")

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
        conclusion_excerpt=_excerpt(diag_text),
        protocol_excerpt="; ".join(mkb_excerpts)[:400],
        protocol_section="Справочник МКБ-10 (RU)",
        source_kind="mkb",
    )


def _kp_title(protocol_matches: list[dict[str, Any]] | None, profile: dict[str, Any]) -> str:
    for m in protocol_matches or []:
        title = (m.get("title") or "").strip()
        if title:
            return title[:100]
    paths = profile.get("paths") or []
    if paths:
        return _basename(str(paths[0]))
    return ""


def _complaints_card(doc: ConsultationDocument, ctx: dict[str, Any]) -> dict[str, Any]:
    text = (doc.sections.complaints or "").strip()
    present = bool(text)
    score = _completeness_score(present, text=text, min_chars=12)
    if not present:
        comment = "Жалобы не описаны."
    elif len(text) < 25:
        comment = f"Жалобы указаны кратко: {_excerpt(text, 140)}"
        score = min(score, 60)
    else:
        comment = f"Жалобы: {_excerpt(text, 200)}"
    if doc.extraction_quality.has_undefined and "undefined" in text.lower():
        score = min(score, 45)
        comment += " В тексте есть незаполненные поля."
    return _card(
        "complaints",
        score_pct=score,
        comment_ru=comment,
        conclusion_excerpt=_excerpt(text, 360),
        source_kind="completeness",
    )


def _anamnesis_card(doc: ConsultationDocument, ctx: dict[str, Any]) -> dict[str, Any]:
    disease = (ctx.get("anamnesis_disease") or "").strip()
    life = (ctx.get("anamnesis_life") or "").strip()
    present = bool(disease or life)
    score = _completeness_score(present, text=disease or life, min_chars=20)
    parts: list[str] = []
    if disease:
        parts.append(f"Анамнез заболевания: {_excerpt(disease, 160)}")
    else:
        parts.append("Анамнез заболевания не описан.")
        score = min(score, 50)
    if life:
        parts.append(f"Анамнез жизни: {_excerpt(life, 120)}")
    else:
        parts.append("Анамнез жизни не указан.")
        if score > 72:
            score -= 10
    return _card(
        "anamnesis",
        score_pct=score,
        comment_ru=" ".join(parts),
        conclusion_excerpt=_excerpt(format_anamnesis_excerpt(ctx), 400),
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
        comment = "Объективный статус не описан или слишком краткий."
    elif len((text or "").strip()) < 30:
        comment = f"Объективный статус краткий: {_excerpt(text, 120)}"
    else:
        comment = f"Объективный статус: {_excerpt(text, 160)}"
    if doc.extraction_quality.has_undefined and block_id == "objective_status":
        comment += " Есть незаполненные поля."
        score = min(score, 50)
    return _card(
        block_id,
        score_pct=score,
        comment_ru=comment,
        conclusion_excerpt=_excerpt(text),
        source_kind="completeness",
    )


def _exams_card(
    doc: ConsultationDocument,
    profile: dict[str, Any],
    ctx: dict[str, Any],
    protocol_matches: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    ranked = rank_kp_items_by_context(
        list(profile.get("diagnostics") or []),
        ctx,
        meta=list(profile.get("diagnostics_meta") or []),
        limit=12,
    )
    required = [r["text"] for r in ranked]
    kz_blob = _kz_exam_blob(doc)
    cite = next(
        (c for c in (profile.get("cites") or []) if c.get("chunk_type") in ("diagnostics", "criteria_block", "table")),
        (profile.get("cites") or [{}])[0] if profile.get("cites") else {},
    )
    if not cite.get("chunk_id") and profile.get("paths"):
        try:
            from clinical_knowledge.kz_chunk_match import best_chunk_for_items

            all_chunks: list[dict[str, Any]] = []
            for pth in (profile.get("paths") or [])[:3]:
                all_chunks.extend(profile.get("chunks_by_path", {}).get(pth) or [])
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
            conclusion_excerpt=_excerpt(kz_blob, 360),
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
            conclusion_excerpt=_excerpt(kz_blob, 360),
            protocol_excerpt=proto_text,
            protocol_section=proto_section,
            protocol_page=str(cite.get("page_from") or ""),
            source_kind="kp",
            protocol_path=cite.get("path") or "",
            chunk_id=cite.get("chunk_id"),
            context_ru=basis,
        )

    pct, found, missing = _coverage_pct(required, kz_blob, meta=ranked)
    findings = [f"✓ {x}" for x in found[:6]]
    gaps = [f"— {x}" for x in missing[:6]]
    if kp_title:
        comment = f"КП «{kp_title}»: в КЗ отражено {len(found)} из {len(required)} рекомендуемых обследований."
    else:
        comment = f"В КЗ отражено {len(found)} из {len(required)} обследований по КП."
    if pick_note:
        comment = f"{pick_note} {comment}"
    if not kz_blob:
        pct = min(pct, 35)
        comment += " Назначения и результаты обследований в КЗ не распознаны."

    return _card(
        "exams",
        score_pct=max(pct, 15) if kz_blob else min(pct, 40),
        comment_ru=comment,
        conclusion_excerpt=_excerpt(kz_blob, 360),
        protocol_excerpt=(proto_header + ": " if proto_header else "") + proto_text,
        protocol_section=proto_section or "Обследование",
        protocol_page=str(cite.get("page_from") or ""),
        source_kind="kp",
        protocol_path=cite.get("path") or "",
        chunk_id=cite.get("chunk_id"),
        findings_ru=findings,
        gaps_ru=gaps,
        context_ru=basis,
    )


def _treatment_card(
    doc: ConsultationDocument,
    profile: dict[str, Any],
    ctx: dict[str, Any],
    protocol_matches: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    pool = list(profile.get("medications") or []) + list(profile.get("treatment") or [])
    ranked = rank_kp_items_by_context(
        pool,
        ctx,
        meta=list(profile.get("medications_meta") or []),
        limit=12,
    )
    required = [r["text"] for r in ranked]
    kz_blob = _kz_treatment_blob(doc)
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
            conclusion_excerpt=_excerpt(kz_blob, 360),
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
            conclusion_excerpt=_excerpt(kz_blob, 360),
            protocol_excerpt=proto_text,
            protocol_section=proto_section,
            protocol_page=str(cite.get("page_from") or ""),
            source_kind="kp",
            protocol_path=cite.get("path") or "",
            chunk_id=cite.get("chunk_id"),
            context_ru=basis,
        )

    pct, found, missing = _coverage_pct(required, kz_blob, meta=ranked)
    findings = [f"✓ {x}" for x in found[:6]]
    gaps = [f"— {x}" for x in missing[:6]]
    if kp_title:
        comment = f"КП «{kp_title}»: в КЗ отражено {len(found)} из {len(required)} рекомендаций по лечению."
    else:
        comment = f"В КЗ отражено {len(found)} из {len(required)} назначений по КП."
    if pick_note:
        comment = f"{pick_note} {comment}"
    if not kz_blob:
        pct = min(pct, 35)
        comment += " Назначения в КЗ не распознаны."

    return _card(
        "treatment",
        score_pct=max(pct, 20) if kz_blob else min(pct, 45),
        comment_ru=comment,
        conclusion_excerpt=_excerpt(kz_blob, 360),
        protocol_excerpt=(proto_header + ": " if proto_header else "") + proto_text,
        protocol_section=proto_section or "Лечение",
        protocol_page=str(cite.get("page_from") or ""),
        source_kind="kp",
        protocol_path=cite.get("path") or "",
        chunk_id=cite.get("chunk_id"),
        findings_ru=findings,
        gaps_ru=gaps,
        context_ru=basis,
    )


def _follow_up_card(
    doc: ConsultationDocument,
    icd_codes: list[str],
    profile: dict[str, Any],
) -> dict[str, Any]:
    follow_text = doc.sections.follow_up_text or ""
    if doc.follow_up:
        follow_text = follow_text or "; ".join(
            (f.raw_text or "") for f in doc.follow_up if f.raw_text
        )
    blob = "\n".join(
        x for x in [
            follow_text,
            doc.sections.general_recommendations or "",
            doc.sections.recommendations_treatment or "",
        ] if x
    )

    reg = lookup_follow_up_expectations(icd_codes)
    kp_mon = list(profile.get("monitoring") or [])
    has_follow = follow_up_mentioned_in_text(blob, min_months=reg.get("min_interval_months"))

    hints = list(reg.get("follow_up_hints") or [])
    if kp_mon:
        hints.append(kp_mon[0][:200])

    proto_excerpt = ""
    if kp_mon:
        proto_excerpt = kp_mon[0][:280]
        source_kind = "kp"
        section = "Диспансерное наблюдение (КП)"
    else:
        proto_excerpt = (reg.get("conclusion_requirement") or hints[0] if hints else "")[:280]
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
        conclusion_excerpt=_excerpt(blob),
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
    profile["chunks_by_path"] = {
        p: get_chunks(p) or [] for p in (profile.get("paths") or protocol_paths or [])[:4]
    }
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
    cards.append(_exams_card(doc, profile, ctx, protocol_matches))
    cards.append(_treatment_card(doc, profile, ctx, protocol_matches))
    cards.append(_follow_up_card(doc, icd_codes, profile))
    cards.append(_limitations_card(profile, ctx, protocol_matches))

    by_id = {c["block_id"]: c for c in cards}
    ordered = [by_id[bid] for bid in ALIGNMENT_CARD_ORDER if bid in by_id]

    scorable = [c for c in ordered if c["block_id"] != "limitations"]
    mean_score = round(sum(c["score_pct"] for c in scorable) / len(scorable)) if scorable else 0

    limitations = " ".join(
        c["comment_ru"] for c in ordered if c["block_id"] == "limitations"
    )

    return {
        "alignment_cards": ordered,
        "criteria": [_card_to_criterion(c) for c in ordered if c["block_id"] != "limitations"],
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
        "findings_ru", "gaps_ru", "context_ru",
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
