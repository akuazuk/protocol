"""Единая сводка протокола для навигатора: выводы по разделам без дублей и обрывков.

Один источник правды для страницы `proto-viewer.html` (и потенциально для карточек
поиска): секционная структура «выводов со смыслом», а не сырые чанки.

Наполнение каждого раздела:
  1. Summary Card - **расширенные** структурные утверждения: препараты с дозой/режимом/
     путём/показанием/противопоказаниями/мониторингом; обследования с уровнем обязательности
     и сроком; красные флаги с тяжестью и действиями; наблюдение/маршрут со сроком и
     действиями; критерии диагноза (обязательные/дополнительные/исключающие).
  2. чистый экстрактор из rich-чанков - предложения по типам чанков, с дедупом и фильтром
     служебного (глоссарий, шифр МКБ, юр/адм обвязка, эхо названия), если карточка разрежена.

Каждый вывод несёт цитату+страницу и, где возможно, подтверждение (grounding) по тексту PDF.
Детерминированно, без LLM.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from clinical_knowledge.extract_quality import (
    best_meaningful_excerpt,
    clean_clinical_text,
    is_legal_admin_text,
    meaningful_clinical_excerpt,
    new_deduper,
    normalize_text,
)

# Разделы навигатора: (brief_id, подпись, builder_kind).
_BRIEF_SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("diagnosis", "Диагноз и критерии", "criteria"),
    ("exams", "Обследования", "exams"),
    ("treatment", "Лечение и препараты", "treatment"),
    ("red_flags", "Красные флаги", "red_flags"),
    ("follow_up", "Наблюдение и маршрут", "follow_up"),
)

# chunk_type / угаданный раздел source_text -> раздел навигатора.
_CHUNK_SECTION_MAP: dict[str, str] = {
    "criteria": "diagnosis",
    "classification": "diagnosis",
    "diagnostics": "exams",
    "treatment": "treatment",
    "drug_list": "treatment",
    "routing": "follow_up",
    "prevention": "follow_up",
}

_SECTION_TITLE_HINTS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("treatment", re.compile(r"лечени|терапи|медикамент|фармак|препарат|хирург", re.I)),
    ("exams", re.compile(r"диагностик|обследован|лаборатор|инструментал", re.I)),
    ("follow_up", re.compile(r"госпитализац|маршрут|направлен|наблюден|диспансер|профилакт", re.I)),
    ("diagnosis", re.compile(r"критери|классификац|диагноз|нозолог", re.I)),
)

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[А-ЯЁA-Z0-9«(])")

_CARE_SETTING_RU: dict[str, str] = {
    "outpatient": "Амбулаторно",
    "inpatient": "Стационар",
    "emergency": "Скорая/неотложная",
    "intensive_care": "Реанимация/ИТ",
    "rehabilitation": "Реабилитация",
    "palliative": "Паллиативная помощь",
}

_SEVERITY_RU: dict[str, str] = {
    "low": "низкая",
    "medium": "средняя",
    "high": "высокая",
    "critical": "критическая",
}

# Слова названия протокола, которые не различают нозологию (для детекции эха названия).
_TITLE_STOP = {
    "клинический", "протокол", "диагностика", "диагностики", "лечение", "лечения",
    "пациент", "пациента", "пациентов", "пациенты", "взрослое", "взрослых", "взрослого",
    "детское", "детей", "детского", "население", "населения", "нас", "медицинской",
    "помощи", "оказании", "оказания", "некоторых", "утверждении",
}


def _title_norm(title: str | None) -> str:
    low = normalize_text(title).lower().replace("ё", "е")
    return re.sub(r"[^а-яa-z0-9 ]+", " ", low).strip()


def _core_tokens(s: str) -> set[str]:
    return {w for w in _title_norm(s).split() if len(w) >= 3 and w not in _TITLE_STOP}


def _is_title_echo(text: str, title_norm: str, title_core: set[str]) -> bool:
    """True, если вывод лишь повторяет название протокола (без клинической сути)."""
    tn = _title_norm(text)
    if not tn:
        return True
    if title_norm and (tn in title_norm or title_norm in tn):
        return True
    tt = _core_tokens(text)
    if not tt:
        return True
    if title_core and tt <= title_core:
        return True
    return False


def _sentences(text: str) -> list[str]:
    t = normalize_text(text)
    if not t:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]


def _guess_brief_section(chunk: dict[str, Any]) -> str | None:
    ctype = str(chunk.get("chunk_type") or chunk.get("kind") or "").strip().lower()
    if ctype in _CHUNK_SECTION_MAP:
        return _CHUNK_SECTION_MAP[ctype]
    title = str(chunk.get("section_title") or chunk.get("title") or "")
    for sec_id, rx in _SECTION_TITLE_HINTS:
        if rx.search(title):
            return sec_id
    return None


# --- Сущности-чипы -----------------------------------------------------------
_ENTITY_NOISE = (
    "пациент", "население", "средств", "назнач", "осуществ", "предусматр",
    "приложени", "продолжа", "обострени", "показани", "терапи", "лечени",
    "может", "или ",
)
_ENTITY_WORD = re.compile(r"[A-Za-zА-Яа-яЁё0-9]")


def _clean_entity(value: Any, *, max_words: int = 3) -> str | None:
    s = re.sub(r"\s+", " ", str(value or "").replace("\n", " ")).strip(" .,;:-")
    if not s or "\n" in s or len(s) > 48:
        return None
    if len(s.split(" ")) > max_words:
        return None
    if any(n in s.lower().replace("ё", "е") for n in _ENTITY_NOISE):
        return None
    if len(_ENTITY_WORD.findall(s)) < 2:
        return None
    return s[:48]


def _dedup_strings(values: list[Any], *, limit: int = 12, max_words: int = 3) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = _clean_entity(v, max_words=max_words)
        if not s:
            continue
        key = s.lower().replace("ё", "е")
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _entities_from_card(cond: Any) -> dict[str, list[str]]:
    """Чипы препаратов/обследований из структурной карточки (чистые названия)."""
    drugs: list[str] = []
    exams: list[str] = []
    tb = getattr(cond, "treatment", None)
    if tb is not None:
        for d in getattr(tb, "drugs", None) or []:
            drugs.append(getattr(d, "drug_name", None) or getattr(d, "active_substance", None) or "")
        for g in getattr(tb, "drug_groups", None) or []:
            drugs.append(getattr(g, "drug_group", None) or "")
    for ex in (getattr(cond, "required_exams", None) or []) + (getattr(cond, "conditional_exams", None) or []):
        exams.append(getattr(ex, "name", None) or "")
    return {"drugs": _dedup_strings(drugs, max_words=4), "exams": _dedup_strings(exams, max_words=4)}


def _entities_from_chunks(rich_chunks: list[dict[str, Any]]) -> dict[str, list[str]]:
    drugs: list[Any] = []
    exams: list[Any] = []
    for ch in rich_chunks or []:
        drugs.extend(ch.get("drugs") or [])
        exams.extend(ch.get("lab_tests") or [])
        exams.extend(ch.get("imaging") or [])
        exams.extend(ch.get("procedures") or [])
    return {"drugs": _dedup_strings(drugs), "exams": _dedup_strings(exams)}


# --- Финализатор точки -------------------------------------------------------
class _Ctx:
    def __init__(self, *, title_norm, title_core, deduper, page_lookup, catalog_path, limit):
        self.title_norm = title_norm
        self.title_core = title_core
        self.deduper = deduper
        self.page_lookup = page_lookup
        self.catalog_path = catalog_path
        self.limit = limit


def _clean_short(s: str | None, limit: int) -> str:
    t = clean_clinical_text(s)
    if not t or len(t) < 3:
        return ""
    if is_legal_admin_text(t):
        return ""
    if len(t) > limit:
        t = t[: limit - 1].rstrip() + "…"
    return t


def _finalize(
    main: str | None,
    *,
    quote: str | None,
    page: Any,
    tags: list[str] | None,
    detail: list[dict[str, str]] | None,
    sentence: bool,
    ctx: _Ctx,
) -> dict[str, Any] | None:
    if sentence:
        text = best_meaningful_excerpt([main, quote], limit=ctx.limit, require_sentence_start=True)
    else:
        text = _clean_short(main, ctx.limit) or best_meaningful_excerpt([quote], limit=ctx.limit)
    if not text or is_legal_admin_text(text):
        return None
    if _is_title_echo(text, ctx.title_norm, ctx.title_core):
        return None
    if not ctx.deduper.accept(text):
        return None
    q = str(quote or "").strip()[:600] or None
    page_source = "summary" if page else None
    grounded = False
    if q and ctx.page_lookup is not None:
        try:
            found = ctx.page_lookup(ctx.catalog_path, q)
        except Exception:
            found = None
        if found:
            grounded = True
            if not page:
                page, page_source = found, "matched"
    verified = grounded if ctx.page_lookup is not None else bool(q)
    return {
        "text": text,
        "quote": q,
        "page_start": page,
        "page_source": page_source,
        "verified": verified,
        "tags": [t for t in (tags or []) if t],
        "detail": [d for d in (detail or []) if d.get("value")],
    }


def _sr(obj: Any) -> tuple[str | None, Any]:
    sr = getattr(obj, "source_ref", None)
    quote = str(getattr(sr, "quote", None) or "").strip() or None
    page = getattr(sr, "page_start", None)
    return quote, page


def _kv(label: str, value: Any) -> dict[str, str]:
    return {"label": label, "value": str(value or "").strip()}


# --- Строители разделов из карточки ------------------------------------------
def _diag_points(cond: Any, *, ctx: _Ctx, max_points: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ds = getattr(cond, "diagnosis_structure", None)
    if ds is not None:
        for comp, tag in (
            [(c, "обязательный") for c in getattr(ds, "required_components", None) or []]
            + [(c, "дополнительный") for c in getattr(ds, "optional_components", None) or []]
        ):
            if len(out) >= max_points:
                break
            q, pg = _sr(comp)
            main = getattr(comp, "name", None) or getattr(comp, "description", None)
            p = _finalize(main, quote=q, page=pg, tags=[tag], detail=[], sentence=False, ctx=ctx)
            if p:
                out.append(p)
    for block in (getattr(cond, "diagnostic_criteria", None), getattr(cond, "clinical_criteria", None)):
        if block is None:
            continue
        groups = (
            [(i, "обязательный") for i in getattr(block, "required", None) or []]
            + [(i, "дополнительный") for i in getattr(block, "optional", None) or []]
            + [(i, "исключающий") for i in getattr(block, "exclusion", None) or []]
        )
        for item, tag in groups:
            if len(out) >= max_points:
                break
            q, pg = _sr(item)
            p = _finalize(getattr(item, "text", None), quote=q, page=pg, tags=[tag], detail=[], sentence=True, ctx=ctx)
            if p:
                out.append(p)
    return out


def _exam_points(cond: Any, *, ctx: _Ctx, max_points: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    req = [(e, "обязательно") for e in getattr(cond, "required_exams", None) or []]
    cond_ex = [(e, "по показаниям") for e in getattr(cond, "conditional_exams", None) or []]
    for ex, tag in req + cond_ex:
        if len(out) >= max_points:
            break
        q, pg = _sr(ex)
        detail = []
        if getattr(ex, "timing", None):
            detail.append(_kv("Когда", ex.timing))
        rif = getattr(ex, "required_if", None) or []
        if rif:
            detail.append(_kv("Если", "; ".join(rif)))
        if getattr(ex, "comment", None):
            detail.append(_kv("Комментарий", ex.comment))
        p = _finalize(getattr(ex, "name", None), quote=q, page=pg, tags=[tag], detail=detail, sentence=False, ctx=ctx)
        if p:
            out.append(p)
    return out


def _drug_detail(d: Any) -> list[dict[str, str]]:
    detail: list[dict[str, str]] = []
    if getattr(d, "route", None):
        detail.append(_kv("Путь", d.route))
    if getattr(d, "dose_text", None):
        detail.append(_kv("Доза", d.dose_text))
    if getattr(d, "frequency_text", None):
        detail.append(_kv("Режим", d.frequency_text))
    if getattr(d, "duration_text", None):
        detail.append(_kv("Длительность", d.duration_text))
    if getattr(d, "indication", None):
        detail.append(_kv("Показание", d.indication))
    ci = getattr(d, "contraindications", None) or []
    if ci:
        detail.append(_kv("Противопоказания", "; ".join(ci)))
    mon = getattr(d, "monitoring", None) or []
    if mon:
        detail.append(_kv("Мониторинг", "; ".join(mon)))
    return detail


def _treatment_points(cond: Any, *, ctx: _Ctx, max_points: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tb = getattr(cond, "treatment", None)
    if tb is None:
        return out
    for d in getattr(tb, "drugs", None) or []:
        if len(out) >= max_points:
            break
        q, pg = _sr(d)
        name = getattr(d, "drug_name", None) or getattr(d, "active_substance", None) or getattr(d, "drug_group", None)
        detail = _drug_detail(d)
        # Если есть структура (доза/режим/...), главная строка = название; иначе - цитата-предложение.
        if detail:
            p = _finalize(name, quote=q, page=pg, tags=["препарат"], detail=detail, sentence=False, ctx=ctx)
        else:
            p = _finalize(name, quote=q, page=pg, tags=["препарат"], detail=[], sentence=False, ctx=ctx)
        if p:
            out.append(p)
    for g in getattr(tb, "drug_groups", None) or []:
        if len(out) >= max_points:
            break
        q, pg = _sr(g)
        detail = [_kv("Показание", getattr(g, "indication", None))] if getattr(g, "indication", None) else []
        p = _finalize(getattr(g, "drug_group", None), quote=q, page=pg, tags=["группа ЛС"], detail=detail, sentence=False, ctx=ctx)
        if p:
            out.append(p)
    for coll, tag in (
        (getattr(tb, "non_drug", None) or [], "немедикаментозно"),
        (getattr(tb, "procedures", None) or [], "процедура"),
        (getattr(tb, "surgery", None) or [], "хирургия"),
    ):
        for item in coll:
            if len(out) >= max_points:
                break
            q, pg = _sr(item)
            main = getattr(item, "text", None) or getattr(item, "name", None)
            detail = [_kv("Показание", getattr(item, "indication", None))] if getattr(item, "indication", None) else []
            p = _finalize(main, quote=q, page=pg, tags=[tag], detail=detail, sentence=True, ctx=ctx)
            if p:
                out.append(p)
    return out


def _red_flag_points(cond: Any, *, ctx: _Ctx, max_points: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rf in getattr(cond, "red_flags", None) or []:
        if len(out) >= max_points:
            break
        q, pg = _sr(rf)
        tags = []
        sev = _SEVERITY_RU.get(str(getattr(rf, "severity", "") or ""))
        if sev:
            tags.append("тяжесть: " + sev)
        detail = []
        acts = getattr(rf, "expected_actions", None) or []
        if acts:
            detail.append(_kv("Действия", "; ".join(acts)))
        p = _finalize(getattr(rf, "text", None), quote=q, page=pg, tags=tags, detail=detail, sentence=True, ctx=ctx)
        if p:
            out.append(p)
    return out


def _follow_up_points(cond: Any, *, ctx: _Ctx, max_points: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seq = (
        [(i, "наблюдение") for i in getattr(cond, "follow_up", None) or []]
        + [(i, "госпитализация") for i in getattr(cond, "hospitalization", None) or []]
        + [(i, "маршрут") for i in getattr(cond, "routing", None) or []]
    )
    for item, tag in seq:
        if len(out) >= max_points:
            break
        q, pg = _sr(item)
        detail = []
        if getattr(item, "timing", None):
            detail.append(_kv("Когда", item.timing))
        acts = getattr(item, "expected_actions", None) or []
        if acts:
            detail.append(_kv("Действия", "; ".join(acts)))
        rif = getattr(item, "required_if", None) or []
        if rif:
            detail.append(_kv("Если", "; ".join(rif)))
        main = getattr(item, "text", None) or getattr(item, "indication", None)
        p = _finalize(main, quote=q, page=pg, tags=[tag], detail=detail, sentence=True, ctx=ctx)
        if p:
            out.append(p)
    return out


_SECTION_BUILDERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "criteria": _diag_points,
    "exams": _exam_points,
    "treatment": _treatment_points,
    "red_flags": _red_flag_points,
    "follow_up": _follow_up_points,
}


def _points_from_chunks(
    rich_chunks: list[dict[str, Any]],
    brief_id: str,
    *,
    limit: int,
    max_points: int,
    ctx: _Ctx,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ch in rich_chunks or []:
        if len(out) >= max_points:
            break
        tags = ch.get("tags") or {}
        if tags.get("is_preamble") or tags.get("signal") == "low":
            continue
        if _guess_brief_section(ch) != brief_id:
            continue
        page = ch.get("page_from") or ch.get("page")
        for sent in _sentences(str(ch.get("text") or "")):
            if is_legal_admin_text(sent):
                continue
            if len(out) >= max_points:
                break
            text = meaningful_clinical_excerpt(sent, limit=limit, require_sentence_start=True)
            if not text or _is_title_echo(text, ctx.title_norm, ctx.title_core):
                continue
            if not ctx.deduper.accept(text):
                continue
            out.append(
                {
                    "text": text,
                    "quote": None,
                    "page_start": page,
                    "page_source": "chunk" if page else None,
                    "verified": False,
                    "tags": [],
                    "detail": [],
                }
            )
    return out


# --- Care setting backfill (P6) ---------------------------------------------
_INPATIENT_RE = re.compile(r"стационар|госпитализ|больничн|в отделени|реаним|интенсивной тер", re.I)
_OUTPATIENT_RE = re.compile(r"амбулатор|поликлин|на дому|диспансерн", re.I)


def _backfill_care_setting(cond: Any, sections_out: list[dict[str, Any]]) -> list[str]:
    blob_parts: list[str] = []
    for item in (getattr(cond, "hospitalization", None) or []) + (getattr(cond, "routing", None) or []):
        blob_parts.append(str(getattr(item, "text", "") or ""))
    for s in sections_out:
        for p in s.get("points", []):
            blob_parts.append(p.get("text", ""))
    blob = " ".join(blob_parts)
    labels: list[str] = []
    if _OUTPATIENT_RE.search(blob):
        labels.append(_CARE_SETTING_RU["outpatient"])
    if _INPATIENT_RE.search(blob):
        labels.append(_CARE_SETTING_RU["inpatient"])
    return labels


def build_protocol_brief(
    catalog_path: str,
    *,
    query: str = "",
    icd_codes: list[str] | None = None,
    rich_chunks: list[dict[str, Any]] | None = None,
    page_lookup: Callable[[str, str], Any] | None = None,
    title_hint: str | None = None,
    max_points: int = 8,
    max_text_chars: int = 320,
    min_points_per_section: int = 3,
) -> dict[str, Any]:
    """Единая расширенная сводка протокола: выводы по разделам + сущности + нозология."""
    from clinical_knowledge.protocol_summary.nav import (
        _condition_nav,
        _condition_obj_by_id,
        dedupe_nav_conditions,
        find_summary_by_catalog_path,
    )

    rich = rich_chunks or []
    summary = find_summary_by_catalog_path(catalog_path)

    title = ""
    protocol_id = ""
    condition_out: dict[str, Any] | None = None
    conditions_total = 0
    care_labels: list[str] = []
    review_status = None
    extraction_status = None
    cond = None

    if summary is not None:
        title = summary.source.title or ""
        protocol_id = summary.protocol_id
        review_status = getattr(summary, "review_status", None)
        extraction_status = getattr(summary, "extraction_status", None)
        care_codes = [cs for cs in (summary.applicability.care_setting or []) if cs and cs != "unknown"]
        care_labels = [_CARE_SETTING_RU[cs] for cs in care_codes if cs in _CARE_SETTING_RU]
        nav_conditions = [_condition_nav(c, query=query, icd_codes=icd_codes) for c in summary.conditions]
        nav_conditions = [c for c in nav_conditions if c.get("sections")]
        nav_conditions = dedupe_nav_conditions(
            nav_conditions,
            query=query,
            icd_codes=icd_codes,
            protocol_title=title,
        )
        conditions_total = len(nav_conditions)
        if nav_conditions:
            top = nav_conditions[0]
            cond = _condition_obj_by_id(summary, top.get("condition_id"), top.get("alias_condition_ids"))
            condition_out = {
                "condition_id": top.get("condition_id"),
                "name": top.get("name"),
                "display_label": top.get("display_label"),
                "icd10_codes": top.get("icd10_codes") or [],
                "icd_match": bool(top.get("icd_match")),
                "name_match": bool(top.get("name_match")),
                "match_reason": top.get("match_reason"),
            }

    if not title:
        title = title_hint or ""
        for ch in rich:
            t = str(ch.get("protocol_title_normalized") or ch.get("protocol_title") or "").strip()
            if t:
                title = title or t
                break

    title_norm = _title_norm(title)
    title_core = _core_tokens(title)

    sections_out: list[dict[str, Any]] = []
    for brief_id, label, kind in _BRIEF_SECTIONS:
        deduper = new_deduper()
        ctx = _Ctx(
            title_norm=title_norm,
            title_core=title_core,
            deduper=deduper,
            page_lookup=page_lookup,
            catalog_path=catalog_path,
            limit=max_text_chars,
        )
        points: list[dict[str, Any]] = []
        if cond is not None:
            builder = _SECTION_BUILDERS[kind]
            points = builder(cond, ctx=ctx, max_points=max_points)
        if len(points) < min_points_per_section and rich:
            need = min_points_per_section - len(points)
            points += _points_from_chunks(rich, brief_id, limit=max_text_chars, max_points=need, ctx=ctx)
        if not points:
            continue
        sections_out.append({"id": brief_id, "label": label, "points": points, "count": len(points)})

    # Сущности-чипы: из карточки (чистые названия) с добивкой из чанков.
    if cond is not None:
        entities = _entities_from_card(cond)
        chunk_ent = _entities_from_chunks(rich)
        for kind in ("drugs", "exams"):
            if len(entities[kind]) < 3:
                seen = {x.lower() for x in entities[kind]}
                for extra in chunk_ent[kind]:
                    if extra.lower() not in seen:
                        entities[kind].append(extra)
                        if len(entities[kind]) >= 12:
                            break
    else:
        entities = _entities_from_chunks(rich)

    # Care setting: backfill, если в карточке пусто (P6).
    if not care_labels and cond is not None:
        care_labels = _backfill_care_setting(cond, sections_out)

    source = "summary" if cond is not None else ("rich_chunks" if rich else None)
    if cond is not None and any(p.get("page_source") == "chunk" for s in sections_out for p in s["points"]):
        source = "mixed"

    # Слабое качество карточки: регекс-разметка или черновик (не LLM) - помечаем для ревью.
    needs_review = str(extraction_status or "").lower() in ("auto_extracted", "draft", "structured_fallback")

    return {
        "available": bool(sections_out),
        "path": catalog_path,
        "protocol_id": protocol_id,
        "title": title,
        "source": source,
        "review_status": review_status,
        "extraction_status": extraction_status,
        "needs_review": needs_review,
        "condition": condition_out,
        "conditions_total": conditions_total,
        "care_setting_labels": care_labels,
        "sections": sections_out,
        "entities": entities,
        "full_text_available": bool(rich),
    }
