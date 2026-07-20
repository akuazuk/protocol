"""Единая сводка протокола для навигатора: выводы по разделам без дублей и обрывков.

Один источник правды для страницы `proto-viewer.html` (и потенциально для карточек
поиска): секционная структура «выводов со смыслом», а не сырые чанки.

Приоритет наполнения каждого раздела:
  1. Summary Card - целые клинические утверждения (критерии/обследование/лечение/
     красные флаги/наблюдение) с дословной цитатой и страницей;
  2. чистый экстрактор из rich-чанков - предложения по типам чанков, с дедупом
     (точные и near-дубли) и фильтром служебного (глоссарий, шифр МКБ, эхо названия).

Сущности (препараты, обследования) агрегируются из rich-чанков и Summary Card.
Детерминированно, без LLM.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from clinical_knowledge.extract_quality import (
    best_meaningful_excerpt,
    is_legal_admin_text,
    meaningful_clinical_excerpt,
    new_deduper,
    normalize_text,
)

# Разделы навигатора: (brief_id, подпись, summary_section_id).
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


def _title_norm(title: str | None) -> str:
    low = normalize_text(title).lower().replace("ё", "е")
    return re.sub(r"[^а-яa-z0-9 ]+", " ", low).strip()


def _is_title_echo(text: str, title_norm: str) -> bool:
    """Отсекает выводы, которые просто повторяют название протокола."""
    if not title_norm:
        return False
    tn = _title_norm(text)
    if not tn:
        return True
    return tn in title_norm or title_norm in tn


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


# Шум в списках сущностей auto-извлечённых карточек (не названия ЛС/обследований).
_ENTITY_NOISE = (
    "пациент", "население", "средств", "назнач", "осуществ", "предусматр",
    "приложени", "продолжа", "обострени", "показани", "терапи", "лечени",
    "может", "или ",
)
_ENTITY_WORD = re.compile(r"[A-Za-zА-Яа-яЁё0-9]")


def _clean_entity(value: Any) -> str | None:
    """Оставляет правдоподобные короткие названия (ЛС/обследования), режет шум."""
    s = re.sub(r"\s+", " ", str(value or "").replace("\n", " ")).strip(" .,;:-")
    if not s or "\n" in s or len(s) > 40:
        return None
    words = s.split(" ")
    if len(words) > 3:
        return None
    low = s.lower().replace("ё", "е")
    if any(n in low for n in _ENTITY_NOISE):
        return None
    if len(_ENTITY_WORD.findall(s)) < 2:
        return None
    return s[:40]


def _dedup_strings(values: list[Any], *, limit: int = 10) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = _clean_entity(v)
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


def _entities_from_chunks(rich_chunks: list[dict[str, Any]]) -> dict[str, list[str]]:
    drugs: list[Any] = []
    exams: list[Any] = []
    for ch in rich_chunks or []:
        drugs.extend(ch.get("drugs") or [])
        exams.extend(ch.get("lab_tests") or [])
        exams.extend(ch.get("imaging") or [])
        exams.extend(ch.get("procedures") or [])
    return {"drugs": _dedup_strings(drugs), "exams": _dedup_strings(exams)}


def _points_from_summary_items(
    items: list[dict[str, Any]],
    *,
    limit: int,
    max_points: int,
    title_norm: str,
    deduper: Any,
    page_lookup: Callable[[str, str], Any] | None,
    catalog_path: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in items:
        if len(out) >= max_points:
            break
        text = best_meaningful_excerpt(
            [it.get("text"), it.get("quote")], limit=limit, require_sentence_start=True
        )
        if not text or _is_title_echo(text, title_norm):
            continue
        if not deduper.accept(text):
            continue
        quote = str(it.get("quote") or "").strip()[:600] or None
        page = it.get("page_start")
        page_source = "summary" if page else None
        if not page and page_lookup is not None:
            try:
                found = page_lookup(catalog_path, quote or text)
            except Exception:
                found = None
            if found:
                page, page_source = found, "matched"
        out.append(
            {
                "text": text,
                "quote": quote,
                "page_start": page,
                "page_source": page_source,
                "verified": bool(quote),
            }
        )
    return out


def _points_from_chunks(
    rich_chunks: list[dict[str, Any]],
    brief_id: str,
    *,
    limit: int,
    max_points: int,
    title_norm: str,
    deduper: Any,
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
        chunk_text = str(ch.get("text") or "")
        page = ch.get("page_from") or ch.get("page")
        # Пофразовый фильтр: юридическую обвязку режем на уровне предложения,
        # чтобы не потерять клиническую фразу в том же чанке (напр. рядом с «шифр по МКБ»).
        for sent in _sentences(chunk_text):
            if is_legal_admin_text(sent):
                continue
            if len(out) >= max_points:
                break
            text = meaningful_clinical_excerpt(sent, limit=limit, require_sentence_start=True)
            if not text or _is_title_echo(text, title_norm):
                continue
            if not deduper.accept(text):
                continue
            out.append(
                {
                    "text": text,
                    "quote": None,
                    "page_start": page,
                    "page_source": "chunk" if page else None,
                    "verified": False,
                }
            )
    return out


def build_protocol_brief(
    catalog_path: str,
    *,
    query: str = "",
    icd_codes: list[str] | None = None,
    rich_chunks: list[dict[str, Any]] | None = None,
    page_lookup: Callable[[str, str], Any] | None = None,
    title_hint: str | None = None,
    max_points: int = 6,
    max_text_chars: int = 300,
    min_points_per_section: int = 3,
) -> dict[str, Any]:
    """Единая сводка протокола: выводы по разделам + сущности + метаданные нозологии."""
    from clinical_knowledge.protocol_summary.nav import (
        _collect_section_excerpt_items,
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
    cond = None

    if summary is not None:
        title = summary.source.title or ""
        protocol_id = summary.protocol_id
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
        # из rich-чанков вытащим нормализованное название, если есть
        for ch in rich:
            t = str(ch.get("protocol_title_normalized") or ch.get("protocol_title") or "").strip()
            if t:
                title = title or t
                break

    title_norm = _title_norm(title)
    sections_out: list[dict[str, Any]] = []
    for brief_id, label, summary_sid in _BRIEF_SECTIONS:
        deduper = new_deduper()
        points: list[dict[str, Any]] = []
        if cond is not None:
            items = _collect_section_excerpt_items(cond, summary_sid, limit=16)
            points = _points_from_summary_items(
                items,
                limit=max_text_chars,
                max_points=max_points,
                title_norm=title_norm,
                deduper=deduper,
                page_lookup=page_lookup,
                catalog_path=catalog_path,
            )
        if len(points) < min_points_per_section and rich:
            # Добиваем раздел только до минимума чистыми клиническими предложениями,
            # а не до max_points сырыми чанками (иначе в разделы лезет правовая шапка).
            need = min_points_per_section - len(points)
            points += _points_from_chunks(
                rich,
                brief_id,
                limit=max_text_chars,
                max_points=need,
                title_norm=title_norm,
                deduper=deduper,
            )
        if not points:
            continue
        sections_out.append(
            {
                "id": brief_id,
                "label": label,
                "points": points,
                "count": len(points),
            }
        )

    entities = _entities_from_chunks(rich)

    source = "summary" if cond is not None else ("rich_chunks" if rich else None)
    if cond is not None and any(p.get("page_source") == "chunk" for s in sections_out for p in s["points"]):
        source = "mixed"

    return {
        "available": bool(sections_out),
        "path": catalog_path,
        "protocol_id": protocol_id,
        "title": title,
        "source": source,
        "condition": condition_out,
        "conditions_total": conditions_total,
        "care_setting_labels": care_labels,
        "sections": sections_out,
        "entities": entities,
        "full_text_available": bool(rich),
    }
