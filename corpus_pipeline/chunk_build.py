"""Семантический чанкинг: подпункты, перечни, блоки показаний."""
from __future__ import annotations

import hashlib
import re
from typing import Any

from .entities_extract import (
    build_embedding_ready_text,
    extract_care_settings,
    extract_conditions_snippets,
    extract_drugs_heuristic,
    extract_durations,
    extract_icd10,
    extract_populations,
    keywords_from_text,
)
from .pdf_extract import span_to_pages

# Нумерация: 1. / 1) / 1.1. / п. 12
POINT_START = re.compile(
    r"(?m)(?:^|\n)\s*(?:\d+(?:\.\d+)*[\.\)]\s+|[пП]\.?\s*\d+[\.\)]\s+)",
)


def _chunk_id(doc_id: str, sec_id: str, idx: int) -> str:
    h = hashlib.sha256(f"{doc_id}:{sec_id}:{idx}".encode()).hexdigest()[:16]
    return f"{doc_id}_{sec_id}_{idx}_{h}"


def split_subpoints(text: str) -> list[str]:
    """Разрез по строкам с нумерацией."""
    if not text or len(text) < 100:
        return [text] if text else []
    parts: list[str] = []
    last = 0
    for m in POINT_START.finditer(text):
        if m.start() > last + 30:
            piece = text[last : m.start()].strip()
            if len(piece) > 40:
                parts.append(piece)
        last = m.start()
    tail = text[last:].strip()
    if len(tail) > 40:
        parts.append(tail)
    if not parts:
        return [text]
    return parts


def build_chunks_for_section(
    doc_id: str,
    full_norm: str,
    page_starts: list[int],
    section: dict[str, Any],
    chunk_type_base: str,
) -> list[dict[str, Any]]:
    stext = section.get("text") or ""
    sec_id = section.get("section_id", "s0")
    sec_path = section.get("section_path") or []
    label = section.get("label") or ""

    subpieces = split_subpoints(stext)
    if len(subpieces) == 1 and len(stext) > 3500:
        # Делим длинный блок по абзацам
        paras = [p.strip() for p in stext.split("\n\n") if len(p.strip()) > 80]
        subpieces = paras if paras else subpieces

    out: list[dict[str, Any]] = []
    base_start = int(section.get("start_char", 0))

    for i, piece in enumerate(subpieces):
        if len(piece) < 40:
            continue
        # Позиция куска в полном тексте секции
        local_idx = stext.find(piece[: min(80, len(piece))])
        if local_idx < 0:
            local_idx = 0
        abs_start = base_start + local_idx
        abs_end = abs_start + len(piece)

        icd = extract_icd10(piece)
        pops = extract_populations(piece)
        care = extract_care_settings(piece)
        drugs = extract_drugs_heuristic(piece)
        conds = extract_conditions_snippets(piece)
        durs = extract_durations(piece)
        kws = keywords_from_text(piece, 25)

        point_nums: list[str] = []
        for m in re.finditer(r"(?:^|\n)\s*((?:\d+\.)+\d+|\d+[\.\)])", piece[:500]):
            point_nums.append(m.group(1).strip())

        pf, pt = span_to_pages(page_starts, len(full_norm), abs_start, abs_end)

        ctype = chunk_type_base
        low = piece.lower()
        if any(x in low for x in ("таблиц", "таблица")):
            ctype = "table_block"
        elif "алгоритм" in low:
            ctype = "algorithm"
        elif any(x in low for x in ("показан", "противопоказ")):
            ctype = "criteria_block"
        elif drugs and len(piece) < 1200:
            ctype = "drug_list"

        out.append(
            {
                "chunk_id": _chunk_id(doc_id, sec_id, i),
                "doc_id": doc_id,
                "section_id": sec_id,
                "section_path": sec_path,
                "chunk_type": ctype,
                "text": piece,
                "page_from": pf,
                "page_to": pt,
                "point_numbers": point_nums[:12],
                "icd10_codes": icd,
                "population": pops,
                "conditions": conds,
                "procedures": [],
                "drugs": drugs,
                "care_setting": care,
                "keywords": kws,
                "durations": durs,
                "embedding_ready_text": build_embedding_ready_text(
                    label, piece, icd, pops
                ),
            }
        )
    return out
