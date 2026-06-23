"""Структурированная выдержка из RAG-чанков для карточек протоколов."""
from __future__ import annotations

import re
from typing import Any

_KIND_ORDER: tuple[tuple[str, str], ...] = (
    ("criteria", "Диагностика"),
    ("diagnostics", "Обследование"),
    ("diagnostic", "Обследование"),
    ("pharmacotherapy", "Лечение"),
    ("drug_list", "Лечение"),
    ("treatment", "Лечение"),
    ("prevention", "Наблюдение"),
    ("dispensary", "Наблюдение"),
    ("routing", "Маршрутизация"),
    ("algorithm", "Алгоритм"),
    ("body", "Протокол"),
    ("general", "Протокол"),
)

_KIND_ALIASES: dict[str, str] = {
    "criteria_block": "criteria",
    "summary_diagnostic_criteria": "criteria",
    "summary_required_exams": "diagnostics",
    "protocol_overview": "body",
}


def _norm_kind(row: dict[str, Any]) -> str:
    raw = str(row.get("chunk_type") or row.get("kind") or "body").strip().lower()
    return _KIND_ALIASES.get(raw, raw)


def _clean_excerpt(text: str, max_len: int = 480) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) <= max_len:
        return t
    cut = t[:max_len]
    sp = cut.rfind(". ")
    if sp > max_len // 2:
        return cut[: sp + 1].strip()
    return cut.rstrip() + "…"


def build_structured_excerpt_for_path(
    retrieval: list[dict[str, Any]],
    path: str,
    *,
    max_section_chars: int = 520,
    max_total_chars: int = 2800,
) -> dict[str, Any]:
    """Склейка лучших чанков по категориям для одного PDF."""
    rows = [r for r in (retrieval or []) if isinstance(r, dict) and str(r.get("path") or "") == path]
    if not rows:
        return {"available": False, "path": path}

    rows.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
    by_kind: dict[str, dict[str, Any]] = {}
    for row in rows:
        kind = _norm_kind(row)
        if kind in by_kind:
            continue
        ex = str(row.get("excerpt") or "").strip()
        if not ex or ex.startswith("Протокол:"):
            continue
        by_kind[kind] = row

    sections: list[dict[str, Any]] = []
    snippets: dict[str, str] = {}
    total = 0

    for kind_key, label in _KIND_ORDER:
        row = by_kind.get(kind_key)
        if not row:
            continue
        ex = _clean_excerpt(str(row.get("excerpt") or ""), max_section_chars)
        if not ex:
            continue
        if total + len(ex) > max_total_chars:
            ex = _clean_excerpt(ex, max(120, max_total_chars - total))
        sec_title = str(row.get("section_title") or label)
        page = row.get("page_from") or row.get("page_start")
        sections.append(
            {
                "kind": kind_key,
                "label": label,
                "section_title": sec_title,
                "text": ex,
                "page_start": page,
            }
        )
        snippets[kind_key] = ex[:200]
        total += len(ex)
        if total >= max_total_chars:
            break

    full_parts = [f"{s['label']}: {s['text']}" for s in sections]
    return {
        "available": bool(sections),
        "path": path,
        "sections": sections,
        "text": "\n\n".join(full_parts)[:max_total_chars],
        "snippets": snippets,
    }


def attach_structured_excerpts(
    payload: dict[str, Any],
    retrieval: list[dict[str, Any]] | None,
    *,
    limit: int = 4,
) -> dict[str, Any]:
    """protocol_excerpts для top-N протоколов в ответе assist."""
    protos: list[dict[str, Any]] = []
    llm = payload.get("llm_json")
    if isinstance(llm, dict):
        raw = llm.get("protocols") or []
        protos = [p for p in raw if isinstance(p, dict)]

    ex_map: dict[str, Any] = {}
    for pr in protos[: max(0, int(limit))]:
        pth = str(pr.get("path") or "").strip()
        if not pth or pth in ex_map:
            continue
        built = build_structured_excerpt_for_path(retrieval or [], pth)
        if built.get("available"):
            ex_map[pth] = built

    if ex_map:
        payload["protocol_excerpts"] = ex_map
    return payload
