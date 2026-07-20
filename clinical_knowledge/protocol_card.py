"""Единая карточка-выдержка протокола для навигатора поиска.

Заменяет сырые обрывки текста в списке протоколов на структурированную выдержку.
Приоритет содержимого (fallback-цепочка):

1. Summary Card - целые клинические утверждения по разделам (критерии, обследование,
   лечение, красные флаги) с цитатой и страницей;
2. структурная RAG-выдержка (`protocol_excerpts`) - лучшие чанки по типам;
3. сырой фрагмент retrieval - крайний случай.

Точное название протокола всегда наверху карточки. Детерминированно, без LLM.
"""
from __future__ import annotations

import re
from typing import Any

_SYNTHETIC_EXCERPT_RE = re.compile(r"^Протокол:\s.*Нозология:", re.S)


def _raw_excerpt_by_path(retrieval: list[dict[str, Any]] | None) -> dict[str, str]:
    """Самый длинный осмысленный excerpt на path (аналог ragExcerptByPath в UI)."""
    out: dict[str, str] = {}
    for r in retrieval or []:
        if not isinstance(r, dict):
            continue
        p = str(r.get("path") or "").strip()
        ex = str(r.get("excerpt") or "").strip()
        if not p or not ex:
            continue
        if _SYNTHETIC_EXCERPT_RE.match(ex):
            continue
        if p not in out or len(ex) > len(out[p]):
            out[p] = ex
    return out


def _card_shell(path: str, source: str | None, *, title: str, care_setting_label: str | None) -> dict[str, Any]:
    return {
        "available": False,
        "path": path,
        "source": source,
        "title": title or "",
        "care_setting_labels": [care_setting_label] if care_setting_label else [],
        "condition": None,
        "conditions_total": 0,
        "extracts": [],
    }


def build_protocol_card(
    path: str,
    *,
    query: str = "",
    icd_codes: list[str] | None = None,
    structured_excerpt: dict[str, Any] | None = None,
    raw_excerpt: str | None = None,
    title_hint: str | None = None,
    care_setting_label: str | None = None,
    max_extracts: int = 4,
    max_text_chars: int = 260,
) -> dict[str, Any]:
    """Собрать карточку-выдержку для одного протокола по fallback-цепочке."""
    from clinical_knowledge.protocol_summary.nav import build_protocol_card_from_summary

    card = build_protocol_card_from_summary(
        path,
        query=query,
        icd_codes=icd_codes,
        max_extracts=max_extracts,
        max_text_chars=max_text_chars,
    )
    if card.get("available"):
        if not card.get("title") and title_hint:
            card["title"] = title_hint
        if care_setting_label and not card.get("care_setting_labels"):
            card["care_setting_labels"] = [care_setting_label]
        return card

    title = card.get("title") or title_hint or ""

    if isinstance(structured_excerpt, dict) and structured_excerpt.get("sections"):
        extracts: list[dict[str, Any]] = []
        for s in structured_excerpt["sections"]:
            if len(extracts) >= max_extracts:
                break
            text = str(s.get("text") or "").strip()
            if not text:
                continue
            extracts.append(
                {
                    "section_id": s.get("kind") or "body",
                    "label": s.get("label") or "Из протокола",
                    "text": text[:max_text_chars],
                    "quote": None,
                    "page_start": s.get("page_start"),
                    "section_title": s.get("section_title"),
                }
            )
        if extracts:
            shell = _card_shell(path, "rag", title=title, care_setting_label=care_setting_label)
            shell["available"] = True
            shell["extracts"] = extracts
            return shell

    raw = (raw_excerpt or "").strip()
    if raw:
        shell = _card_shell(path, "raw", title=title, care_setting_label=care_setting_label)
        shell["available"] = True
        shell["extracts"] = [
            {
                "section_id": "body",
                "label": "Фрагмент",
                "text": raw[:400],
                "quote": None,
                "page_start": None,
                "section_title": None,
            }
        ]
        return shell

    return _card_shell(path, None, title=title, care_setting_label=care_setting_label)


def attach_protocol_cards(
    payload: dict[str, Any],
    retrieval: list[dict[str, Any]] | None,
    *,
    query: str = "",
    icd_codes: list[str] | None = None,
    limit: int = 12,
) -> dict[str, Any]:
    """Вложить protocol_card для протоколов из выдачи (все перечисленные, до limit)."""
    protos: list[dict[str, Any]] = []
    llm = payload.get("llm_json")
    if isinstance(llm, dict):
        raw = llm.get("protocols") or []
        protos = [p for p in raw if isinstance(p, dict)]
    if not protos:
        return payload

    codes = list(icd_codes or [])
    if not codes:
        icd_payload = payload.get("icd") or {}
        if isinstance(icd_payload, dict):
            codes = list(icd_payload.get("codes_for_retrieval") or [])

    excerpts = payload.get("protocol_excerpts") or {}
    ui_meta = payload.get("protocol_ui_meta") or {}
    raw_map = _raw_excerpt_by_path(retrieval)

    cards: dict[str, Any] = {}
    for pr in protos[: max(0, int(limit))]:
        path = str(pr.get("path") or "").strip()
        if not path or path in cards:
            continue
        care_label = (ui_meta.get(path) or {}).get("care_setting_label")
        card = build_protocol_card(
            path,
            query=query,
            icd_codes=codes or None,
            structured_excerpt=excerpts.get(path),
            raw_excerpt=raw_map.get(path),
            title_hint=pr.get("title"),
            care_setting_label=care_label,
        )
        if card.get("available"):
            cards[path] = card

    if cards:
        payload["protocol_card"] = cards
    return payload
