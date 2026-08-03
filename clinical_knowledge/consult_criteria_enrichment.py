"""Обогащение карточек criteria: цитаты КЗ, эталоны КП, комментарии, уверенность."""
from __future__ import annotations

import os
import re
from typing import Any, Callable

from clinical_knowledge.consult_evidence_quality import (
    is_usable_evidence_excerpt,
)
from clinical_knowledge.consult_schema import ConsultationDocument
from clinical_knowledge.kz_clinical_context import format_anamnesis_excerpt

from clinical_knowledge.meaningful_excerpt import meaningful_excerpt
from clinical_knowledge.semantic_rule_fallback import fuzzy_term_in_text

import icd_mkb

GetChunksFn = Callable[[str], list[dict[str, Any]]]

_PEDiatric_ONLY = re.compile(
    r"новорожд|детск|реб[её]н|педиатр|грудн|младенц|подростк",
    re.I,
)
_ADULT_ONLY = re.compile(
    r"взросл|беременн|пожил|65\s*лет|старше\s*18",
    re.I,
)
_MED_SHORT = re.compile(
    r"узи|кт\b|мрт|экг|ривароксабан|апиксабан|варфарин|антибиот|колоноскоп|"
    r"фгдс|эгдс|спиромет|холтер|коагул|оак\b",
    re.I,
)
_MATCH_LABEL = {
    "found": "точное",
    "partial": "частичное",
    "entity": "по сущности",
    "alias": "по синониму",
    "fuzzy": "по совпадению",
    "fuzzy_head": "по началу формулировки",
    "short": "по аббревиатуре",
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def expand_kz_blob(doc: ConsultationDocument, kind: str) -> str:
    """Расширенный текст КЗ для сверки с КП (секции + raw_text + лекарства)."""
    s = doc.sections
    parts: list[str] = []
    if kind == "exams":
        parts.extend([
            s.recommendations_exams or "",
            s.exam_results or "",
            s.general_recommendations or "",
            s.recommendations_treatment or "",
        ])
        for ex in doc.performed_exams or []:
            parts.append(getattr(ex, "raw_text", None) or ex.exam_name or "")
    elif kind == "treatment":
        parts.extend([
            s.recommendations_treatment or "",
            s.general_recommendations or "",
        ])
        for m in doc.medications or []:
            parts.append(m.raw_text or m.drug_name or "")
            if getattr(m, "generic_name", None):
                parts.append(str(m.generic_name))
    elif kind == "follow_up":
        parts.extend([
            s.follow_up_text or "",
            s.general_recommendations or "",
            s.recommendations_treatment or "",
        ])
        for f in doc.follow_up or []:
            parts.append(f.raw_text or "")
    else:
        parts.append(s.complaints or "")

    raw = (getattr(doc, "raw_text", None) or "").strip()
    if raw and len(raw) < 12000:
        parts.append(raw)

    return "\n".join(p for p in parts if (p or "").strip())


def section_text_for_block(
    doc: ConsultationDocument,
    block_id: str,
    ctx: dict[str, Any] | None = None,
) -> str:
    """Полный текст раздела КЗ для колонки «Фрагмент КЗ» (без обрезки)."""
    s = doc.sections
    ctx = ctx or {}
    if block_id == "diagnosis":
        diag = (s.diagnosis_text or "").strip()
        if not diag and doc.diagnoses:
            diag = "; ".join(
                (d.diagnosis_name or d.raw_text or "").strip()
                for d in doc.diagnoses
                if (d.diagnosis_name or d.raw_text)
            )
        return diag
    if block_id == "complaints":
        return (s.complaints or "").strip()
    if block_id == "anamnesis":
        return format_anamnesis_excerpt(ctx, limit=4000).strip()
    if block_id == "objective_status":
        parts = [s.objective_status or "", s.local_status or ""]
        return "\n".join(p.strip() for p in parts if p and p.strip())
    if block_id == "exams":
        parts = [
            s.recommendations_exams or "",
            s.exam_results or "",
            s.general_recommendations or "",
        ]
        for ex in doc.performed_exams or []:
            parts.append(getattr(ex, "raw_text", None) or ex.exam_name or "")
        return "\n".join(p.strip() for p in parts if p and str(p).strip())
    if block_id == "treatment":
        parts = [s.recommendations_treatment or ""]
        for m in doc.medications or []:
            parts.append(m.raw_text or m.drug_name or "")
        return "\n".join(p.strip() for p in parts if p and str(p).strip())
    if block_id == "follow_up":
        parts = [s.follow_up_text or "", s.general_recommendations or ""]
        for f in doc.follow_up or []:
            parts.append(f.raw_text or "")
        return "\n".join(p.strip() for p in parts if p and str(p).strip())
    return ""


def kz_source_label(source_file: str) -> str:
    from clinical_knowledge.protocol_links import beautify_protocol_title

    label = beautify_protocol_title(source_file)
    return label or "клиническое заключение"


def _strip_list_prefix(line: str) -> str:
    return re.sub(r"^[✓—\-–\s]+", "", (line or "").strip())


def comment_from_findings_gaps(
    block_id: str,
    findings: list[str],
    gaps: list[str],
    *,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Краткий вывод методисту; детали - в списках ниже."""
    clean_g = [_strip_list_prefix(x) for x in (gaps or []) if x]
    clean_f = [_strip_list_prefix(x) for x in (findings or []) if x]
    parts: list[str] = []
    if prefix:
        parts.append(prefix.strip().rstrip("."))

    if clean_g:
        main = clean_g[0]
        if block_id == "complaints":
            parts.append(f"Дополните жалобы: {main.lower().rstrip('.')}.")
        elif block_id == "anamnesis":
            parts.append(f"В анамнезе не хватает: {main.lower().rstrip('.')}.")
        elif block_id == "objective_status":
            parts.append(f"В объективном статусе: {main.lower().rstrip('.')}.")
        elif block_id == "diagnosis":
            parts.append(f"По диагнозу: {main.rstrip('.')}.")
        elif block_id == "follow_up":
            parts.append(f"По наблюдению: {main.lower().rstrip('.')}.")
        else:
            parts.append(f"Замечание: {main.rstrip('.')}.")
        if len(clean_g) > 1:
            parts.append(f"Также: {'; '.join(clean_g[1:3]).rstrip('.')}.")
    elif clean_f:
        verdict = {
            "complaints": "Жалобы оформлены по требованиям СОП.",
            "anamnesis": "Анамнез заполнен по требованиям СОП.",
            "objective_status": "Объективный статус оформлен по требованиям СОП.",
            "diagnosis": "Диагноз оформлен корректно.",
            "follow_up": "Сроки наблюдения указаны.",
        }
        parts.append(verdict.get(block_id, "Замечаний по разделу нет."))
    else:
        parts.append("Недостаточно данных для вывода по разделу.")

    if suffix:
        parts.append(suffix.strip().rstrip("."))
    return " ".join(parts)


def kp_coverage_comment(
    kp_title: str,
    found: list[str],
    missing: list[str],
    details: list[dict[str, Any]],
    *,
    kind: str = "обследований",
    pick_note: str = "",
) -> str:
    """Вердикт по сверке с КП - без дублирования списков пунктов."""
    from clinical_knowledge.protocol_links import beautify_protocol_title

    pretty_kp = beautify_protocol_title(kp_title) if kp_title else ""
    req_missing = [d for d in details if not d.get("matched") and d.get("obligation") == "required"]
    total = len(found) + len(missing)
    parts: list[str] = []

    if pick_note:
        parts.append(pick_note.strip().rstrip("."))

    if not total:
        if pretty_kp:
            parts.append(f"По протоколу «{pretty_kp}» пункты для сверки не извлечены.")
        else:
            parts.append("Пункты протокола для сверки не извлечены.")
        return ". ".join(parts) + "."

    if req_missing:
        from clinical_knowledge.consult_evidence_quality import is_usable_evidence_excerpt

        names = [
            str(d.get("text") or "")[:80]
            for d in req_missing
            if is_usable_evidence_excerpt(str(d.get("text") or ""))
        ][:3]
        head = f"«{pretty_kp}»" if pretty_kp else "Протокол"
        if names:
            parts.append(
                f"{head}: в заключении не отражены обязательные {kind}: "
                + "; ".join(names)
                + "."
            )
        else:
            parts.append(
                f"{head}: не отражены обязательные {kind} "
                f"({len(req_missing)} пункт(ов)) - сверьте формулировки в КЗ."
            )
    elif missing:
        head = f"«{pretty_kp}»" if pretty_kp else "По протоколу"
        parts.append(
            f"{head}: отражено {len(found)} из {total} {kind}; "
            f"рекомендуемые без отметки в КЗ - {len(missing)}."
        )
    else:
        head = f"«{pretty_kp}»" if pretty_kp else "Назначения"
        parts.append(f"{head}: отражены все проверенные пункты ({len(found)} из {total}).")

    partial = [d for d in details if d.get("matched") and d.get("kz_match") == "partial"]
    if partial:
        parts.append(
            f"Частичное совпадение по {len(partial)} пункту(ам) - уточните формулировки в КЗ."
        )
    return " ".join(parts)


def filter_kp_items_by_demographics(
    items: list[str],
    meta: list[dict[str, Any]] | None,
    ctx: dict[str, Any],
) -> list[str]:
    """Исключить пункты КП, явно не для возраста/пола пациента."""
    age = ctx.get("age_years")
    adult_child = ctx.get("adult_or_child")
    is_child = adult_child == "child" or (isinstance(age, (int, float)) and age < 18)
    is_adult = adult_child == "adult" or (isinstance(age, (int, float)) and age >= 18)

    meta_by: dict[str, dict[str, Any]] = {}
    for m in meta or []:
        if isinstance(m, dict):
            meta_by[str(m.get("text") or "")[:80].lower()] = m

    out: list[str] = []
    for it in items:
        t = (it or "").strip()
        if not t:
            continue
        low = t.lower()
        if is_child and _ADULT_ONLY.search(low) and not _PEDiatric_ONLY.search(low):
            continue
        if is_adult and _PEDiatric_ONLY.search(low) and not _ADULT_ONLY.search(low):
            continue
        out.append(t)
    return out


def kz_evidence_snippet(
    kz_blob: str,
    item: str,
    *,
    raw_text: str = "",
    limit: int = 220,
) -> str:
    """Выдержка из КЗ вокруг совпадения с пунктом КП."""
    for blob in (kz_blob, raw_text):
        if not blob:
            continue
        ok, _, matched = fuzzy_term_in_text(blob, item)
        if not ok and item:
            head = re.split(r"[—\-–;]", item)[0].strip()
            if len(head) >= 8:
                ok, _, matched = fuzzy_term_in_text(blob, head)
        if not ok:
            low = blob.lower()
            for m in _MED_SHORT.finditer(item or ""):
                needle = m.group(0).lower()
                if needle in low:
                    matched = needle
                    ok = True
                    break
        if ok and matched:
            pos = blob.lower().find(matched.lower())
            if pos < 0:
                pos = blob.lower().find((item or "")[:12].lower())
            if pos >= 0:
                start = max(0, pos - 60)
                end = min(len(blob), pos + len(matched or item) + 100)
                snippet = blob[start:end].strip()
                if start > 0:
                    snippet = "…" + snippet
                if end < len(blob):
                    snippet = snippet + "…"
                return meaningful_excerpt(snippet, limit=limit) or snippet[:limit]
    return ""


def _match_confidence_label(km: dict[str, Any]) -> str:
    method = str(km.get("kz_match_method") or "")
    match = str(km.get("kz_match") or "")
    if match == "found" and method in ("fuzzy", "entity"):
        return _MATCH_LABEL.get(method, "по совпадению")
    if match == "partial":
        return _MATCH_LABEL.get(method, "частичное")
    if method == "short":
        return _MATCH_LABEL["short"]
    return _MATCH_LABEL.get(method, "по совпадению")


def coverage_with_evidence(
    required: list[str],
    kz_blob: str,
    *,
    meta: list[dict[str, Any]] | None = None,
    raw_text: str = "",
) -> tuple[int, list[str], list[str], list[dict[str, Any]]]:
    """Сверка пунктов КП с КЗ + детали для UI (уверенность, цитаты)."""
    from clinical_knowledge.kz_chunk_match import match_kp_item_to_kz

    if not required:
        return 0, [], [], []

    found: list[str] = []
    missing: list[str] = []
    details: list[dict[str, Any]] = []
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
        km = match_kp_item_to_kz(item, kz_blob, entities=entities, raw_text=raw_text)
        mentioned = km["kz_match"] in ("found", "partial")
        conf_label = _match_confidence_label(km)
        snippet = kz_evidence_snippet(kz_blob, item, raw_text=raw_text) if mentioned else ""

        detail = {
            "text": item,
            "obligation": obligation,
            "kz_match": km["kz_match"],
            "kz_match_method": km.get("kz_match_method"),
            "confidence": km.get("confidence"),
            "confidence_label": conf_label,
            "kz_snippet": snippet,
            "matched": mentioned,
        }
        details.append(detail)

        if mentioned:
            found.append(item)
            w = 1.0 if km["kz_match"] == "found" else 0.85
            if float(km.get("confidence") or 0) < 0.7:
                w *= 0.9
            got_weight += weight * w
        else:
            missing.append(item)

    pct = round(100 * got_weight / total_weight) if total_weight else 0
    return pct, found, missing, details


def score_from_sop_findings_gaps(
    findings: list[str],
    gaps: list[str],
    *,
    base: int | None = None,
) -> int:
    """Пересчёт % из SOP findings/gaps (после merge_sop_into_card)."""
    f = len(findings or [])
    g = len(gaps or [])
    if f + g == 0:
        return max(0, min(100, int(base or 70)))
    raw = round(100 * f / (f + g))
    if base is not None:
        raw = round(0.55 * raw + 0.45 * base)
    return max(0, min(100, raw))


def findings_gaps_from_details(
    details: list[dict[str, Any]],
    *,
    limit: int = 6,
) -> tuple[list[str], list[str]]:
    """Списки ✓/ - с меткой уверенности."""
    findings: list[str] = []
    gaps: list[str] = []
    for d in details[:12]:
        text = str(d.get("text") or "")
        if not text:
            continue
        if d.get("matched"):
            label = d.get("confidence_label") or "найдено"
            findings.append(f"✓ {text} ({label})")
        else:
            obl = "обяз." if d.get("obligation") == "required" else "рек."
            gaps.append(f" - {text} ({obl})")
    return findings[:limit], gaps[:limit]


def best_kz_excerpt_from_details(
    details: list[dict[str, Any]],
    fallback_blob: str,
    *,
    prefer_matched: bool = True,
) -> str:
    """Фрагмент КЗ: сначала из доказательства match, иначе из blob."""
    if prefer_matched:
        for d in details:
            sn = (d.get("kz_snippet") or "").strip()
            if sn and d.get("matched"):
                return sn
    return meaningful_excerpt(fallback_blob, limit=360) or ""


def verify_protocol_excerpt(
    text: str,
    *,
    cite: dict[str, Any] | None = None,
) -> str:
    """Проверить пригодность эталона; иначе fallback на страницу КП."""
    t = (text or "").strip()
    if t and is_usable_evidence_excerpt(t):
        return meaningful_excerpt(t, limit=360) or t[:360]
    cite = cite or {}
    page = cite.get("page_from")
    path = cite.get("path") or ""
    fname = path.replace("\\", "/").split("/")[-1] if path else ""
    if fname and page:
        return f"Цитата недоступна; см. КП «{fname[:60]}», стр. {page}."
    if fname:
        return f"Цитата недоступна; см. КП «{fname[:60]}»."
    return ""


def gap_protocol_refs(
    gaps: list[str],
    ranked: list[dict[str, Any]],
    cites: list[dict[str, Any]],
    *,
    get_chunks: GetChunksFn | None = None,
    protocol_paths: list[str] | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Эталон КП для каждого gap (топ-N)."""
    ranked_by: dict[str, dict[str, Any]] = {
        str(r.get("text") or "")[:80].lower(): r for r in ranked
    }
    refs: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    if get_chunks and protocol_paths:
        try:
            from clinical_knowledge.consult_memory import cap_chunks_for_consult

            for pth in protocol_paths[:3]:
                all_chunks.extend(cap_chunks_for_consult(get_chunks(pth) or []))
        except Exception:
            pass

    for gap in gaps[:limit]:
        key = gap[:80].lower()
        row = ranked_by.get(key, {"text": gap})
        text = str(row.get("text") or gap)
        cite = next((c for c in cites if text[:40].lower() in (c.get("text") or "").lower()), None)
        chunk_text = ""
        chunk_meta: dict[str, Any] = {}
        if not cite and all_chunks:
            for ch in all_chunks:
                if text[:30].lower() in (ch.get("text") or "").lower():
                    cite = {
                        "path": ch.get("source_path"),
                        "chunk_id": ch.get("chunk_id"),
                        "page_from": ch.get("page_from"),
                        "text": ch.get("text"),
                        "section_title": ch.get("section_title"),
                    }
                    break
        if cite:
            chunk_text = verify_protocol_excerpt(cite.get("text") or "", cite=cite)
            chunk_meta = {
                "protocol_path": cite.get("path") or "",
                "chunk_id": cite.get("chunk_id"),
                "protocol_page": str(cite.get("page_from") or ""),
                "protocol_section": cite.get("section_title") or "КП",
            }
        else:
            chunk_text = verify_protocol_excerpt(text)
        refs.append({
            "gap_text": gap,
            "protocol_excerpt": chunk_text or text[:200],
            **chunk_meta,
        })
    return refs


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


def diagnosis_assessment_lines(doc: ConsultationDocument) -> tuple[list[str], list[int], list[str]]:
    """Расширенная оценка диагноза: МКБ, подрубрики, формулировка."""
    comments: list[str] = []
    scores: list[int] = []
    mkb_excerpts: list[str] = []
    diag_text = doc.sections.diagnosis_text or ""

    for d in doc.diagnoses:
        code = d.icd10_code
        name = d.diagnosis_name or d.raw_text or diag_text
        if not code:
            comments.append("Код МКБ-10 не указан для одного из диагнозов.")
            scores.append(40)
            continue
        valid = icd_mkb.is_code_in_ru_reference(code)
        title = icd_mkb.ru_title(code)
        desc = icd_mkb.describe_code(code)
        if valid and title:
            mkb_excerpts.append(_mkb_line(code, title))
            match = _title_match_score(name, title)
            if match >= 0.35:
                scores.append(95)
                comments.append(
                    f"Код {code} в справочнике МКБ; формулировка согласуется с рубрикой «{title[:80]}»."
                )
            else:
                scores.append(78)
                comments.append(
                    f"Код {code} валиден ({title[:60]}), но текст диагноза слабо совпадает с рубрикой МКБ."
                )
            if re.search(r"\.\d", code) and match < 0.25:
                comments.append(
                    f"Уточните подрубрику МКБ: указан {code}, формулировка не отражает детализацию рубрики."
                )
                scores[-1] = min(scores[-1], 72)
            elif code.endswith(".9") or code.endswith(".0"):
                parent = code.rsplit(".", 1)[0]
                if parent and icd_mkb.ru_title(parent):
                    dt = _norm_tokens(name)
                    pt = _norm_tokens(icd_mkb.ru_title(parent) or "")
                    if dt & pt and match < 0.3:
                        comments.append(
                            f"Код {code} - неуточнённая рубрика; при возможности укажите более специфичный код."
                        )
                        scores[-1] = min(scores[-1], 80)
        elif valid:
            scores.append(72)
            comments.append(f"Код {code} найден в справочнике МКБ.")
            if desc.get("title_ru"):
                mkb_excerpts.append(_mkb_line(code, str(desc["title_ru"])))
        else:
            scores.append(30)
            comments.append(f"Код {code} не найден в русском справочнике МКБ-10.")

    return comments, scores, mkb_excerpts


def _mkb_line(code: str, title: str | None) -> str:
    c = (code or "").strip()
    t = (title or "").strip()
    if not c:
        return t
    if not t:
        return c
    if t.upper().startswith(c.upper()):
        return t
    return f"{c} - {t}"


def finalize_completeness_card(card: dict[str, Any]) -> None:
    """После SOP: пересчитать % и комментарий для completeness-блоков."""
    bid = str(card.get("block_id") or "")
    if card.get("source_kind") != "completeness" and bid not in (
        "complaints", "anamnesis", "objective_status",
    ):
        return
    findings = list(card.get("findings_ru") or [])
    gaps = list(card.get("gaps_ru") or [])
    base = int(card.get("score_pct") or 0)
    card["score_pct"] = score_from_sop_findings_gaps(findings, gaps, base=base)
    card["comment_ru"] = comment_from_findings_gaps(bid, findings, gaps)
    if not (card.get("conclusion_excerpt") or "").strip():
        card["conclusion_excerpt"] = card.get("conclusion_excerpt") or ""
    ref = (card.get("reference_ru") or card.get("protocol_excerpt") or "").strip()
    if ref:
        card["protocol_excerpt"] = ref
        if not card.get("protocol_section"):
            card["protocol_section"] = "СОП № 2 Кравира (амбулаторная карта)"


def enrich_kp_card(
    card: dict[str, Any],
    *,
    details: list[dict[str, Any]],
    kz_blob: str,
    cite: dict[str, Any],
    ranked: list[dict[str, Any]],
    get_chunks: GetChunksFn | None = None,
    protocol_paths: list[str] | None = None,
) -> None:
    """Дополнить KP-карточку: gap_refs, evidence excerpt, verified protocol."""
    findings, gaps = findings_gaps_from_details(details)
    card["findings_ru"] = findings
    card["gaps_ru"] = gaps
    card["item_details"] = details[:12]
    if not (card.get("conclusion_excerpt") or "").strip():
        card["conclusion_excerpt"] = best_kz_excerpt_from_details(details, kz_blob) or ""

    proto = verify_protocol_excerpt(cite.get("text") or "", cite=cite)
    if proto:
        card["protocol_excerpt"] = proto
    card["protocol_section"] = cite.get("section_title") or card.get("protocol_section") or "КП"
    card["protocol_page"] = str(cite.get("page_from") or card.get("protocol_page") or "")
    card["protocol_path"] = cite.get("path") or card.get("protocol_path") or ""
    card["chunk_id"] = cite.get("chunk_id") or card.get("chunk_id")
    try:
        from clinical_knowledge.protocol_links import protocol_display_name

        ppath = str(cite.get("path") or card.get("protocol_path") or "")
        if ppath:
            card["protocol_title"] = protocol_display_name(ppath)
    except Exception:
        pass

    gap_items = [str(d.get("text") or "") for d in details if not d.get("matched")]
    card["gap_protocol_refs"] = gap_protocol_refs(
        gap_items,
        ranked,
        list(card.get("_cites") or []),
        get_chunks=get_chunks,
        protocol_paths=protocol_paths,
    )


def maybe_apply_criteria_narrative(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Опционально: LLM перефразирует comment_ru без изменения фактов."""
    if not _env_bool("CONSULT_CRITERIA_NARRATIVE", False):
        return criteria
    try:
        from clinical_knowledge.consult_criteria_narrative import enrich_criteria_comments_llm

        return enrich_criteria_comments_llm(criteria)
    except Exception:
        return criteria
