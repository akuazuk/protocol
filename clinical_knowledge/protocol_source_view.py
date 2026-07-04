"""Компактное клиническое представление source_text для UI-навигатора протокола."""
from __future__ import annotations

import re
from typing import Any

from clinical_knowledge.protocol_summary.source_text import _SECTION_LABELS_RU, _SECTION_ORDER

_VIEW_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("diagnosis", "Диагноз и классификация", ("classification", "criteria")),
    ("diagnostics", "Диагностика и обследования", ("diagnostics",)),
    ("treatment", "Лечение и терапия", ("treatment",)),
    ("followup", "Наблюдение и показания", ("prevention", "routing")),
)

_TITLE_NOISE = re.compile(
    r"^(?:документ|описание протокола|постановляет|согласовано|утверждено|"
    r"клинический протокол|глава\s*\d|постановление министерства|"
    r"национальный правовой|№\s*\d)",
    re.I,
)
_TABLE_MARK = re.compile(r"\|\s*---\s*\|")
_BROKEN_LINE = re.compile(r"(?<=[а-яёa-z])\n(?=[а-яёa-z])", re.I)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def _normalize_block_text(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").strip()
    t = _BROKEN_LINE.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    lines = [ln.strip() for ln in t.split("\n")]
    merged: list[str] = []
    buf = ""
    for ln in lines:
        if not ln:
            if buf:
                merged.append(buf)
                buf = ""
            continue
        if buf and len(buf) < 80 and not buf.endswith((".", "!", "?", ";", ":")):
            buf = f"{buf} {ln}"
        else:
            if buf:
                merged.append(buf)
            buf = ln
    if buf:
        merged.append(buf)
    out = "\n".join(merged)
    return _MULTI_SPACE.sub(" ", out).strip()


def _title_usable(title: str | None) -> bool:
    t = (title or "").strip()
    if len(t) < 8:
        return False
    if _TITLE_NOISE.search(t):
        return False
    if t.endswith(("-", "–", "—")):
        return False
    tokens = t.split()
    if len(tokens) <= 2 and len(t) < 20:
        return False
    return True


def _block_fingerprint(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())[:200]


def _is_garbage_block(text: str, title: str | None = None) -> bool:
    t = (text or "").strip()
    if len(t) < 20:
        return True
    if _TABLE_MARK.search(t) or t.count("|") >= 6:
        return True
    if _title_usable(title) and _TITLE_NOISE.search(title or ""):
        return True
    try:
        from clinical_knowledge.chunk_tags import is_administrative_text
        from clinical_knowledge.consult_evidence_quality import (
            clean_clinical_sentences,
            is_reference_noise,
            is_usable_evidence_excerpt,
        )

        if is_administrative_text(t):
            return True
        if is_reference_noise(t):
            return True
        if not is_usable_evidence_excerpt(t) and clean_clinical_sentences(t) is None:
            return True
    except Exception:
        pass
    return False


def _item_lead(text: str, title: str | None) -> str | None:
    try:
        from clinical_knowledge.consult_evidence_quality import clean_clinical_sentences

        lead = clean_clinical_sentences(text, max_sentences=1, max_chars=140)
        if lead:
            return lead
    except Exception:
        pass
    norm = _normalize_block_text(text)
    if _title_usable(title):
        tt = re.sub(r"\s+", " ", title or "").strip()
        if len(tt) <= 120:
            return tt
    first = norm.split("\n", 1)[0].strip()
    if len(first) >= 20:
        return first[:140] + ("…" if len(first) > 140 else "")
    return None


def _item_body(text: str) -> str | None:
    try:
        from clinical_knowledge.consult_evidence_quality import clean_clinical_sentences

        body = clean_clinical_sentences(text, max_sentences=4, max_chars=520)
        if body:
            return body
    except Exception:
        pass
    norm = _normalize_block_text(text)
    if len(norm) < 24:
        return None
    if len(norm) > 520:
        return norm[:519].rstrip() + "…"
    return norm


def prepare_protocol_source_view(doc: dict[str, Any], *, max_per_group: int = 10) -> dict[str, Any]:
    """Сжимает source_text: без дублей, служебного шума и сырых переносов PDF."""
    raw_sections = doc.get("sections") or {}
    seen: set[str] = set()
    view_sections: dict[str, list[dict[str, Any]]] = {}
    toc: list[dict[str, Any]] = []
    raw_blocks = 0
    filtered_blocks = 0

    for group_id, group_label, source_keys in _VIEW_GROUPS:
        items: list[dict[str, Any]] = []
        for key in source_keys:
            for block in raw_sections.get(key) or []:
                if not isinstance(block, dict):
                    continue
                raw_blocks += 1
                title = str(block.get("section_title") or "").strip() or None
                text = str(block.get("text") or "")
                norm = _normalize_block_text(text)
                if not norm:
                    filtered_blocks += 1
                    continue
                if _is_garbage_block(norm, title):
                    filtered_blocks += 1
                    continue
                fp = _block_fingerprint(norm)
                if fp in seen:
                    filtered_blocks += 1
                    continue
                lead = _item_lead(norm, title)
                body = _item_body(norm)
                if not lead or not body:
                    filtered_blocks += 1
                    continue
                seen.add(fp)
                page = block.get("page_from") or block.get("page")
                items.append(
                    {
                        "id": f"{group_id}-{len(items)}",
                        "lead": lead,
                        "body": body,
                        "page": page,
                        "source_section": key,
                        "source_section_label": _SECTION_LABELS_RU.get(key, key),
                    }
                )
                if len(items) >= max_per_group:
                    break
            if len(items) >= max_per_group:
                break
        if items:
            view_sections[group_id] = items
            toc.append({"id": group_id, "label": group_label, "count": len(items)})

    return {
        "toc": toc,
        "sections": view_sections,
        "stats": {
            "raw_blocks": raw_blocks,
            "shown_blocks": sum(len(v) for v in view_sections.values()),
            "filtered_blocks": filtered_blocks,
        },
        "section_labels": {gid: lbl for gid, lbl, _ in _VIEW_GROUPS},
    }
