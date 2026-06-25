"""Evidence pack для L2: секционные выдержки из summary cards и rich chunks."""
from __future__ import annotations

from typing import Any, Callable

from clinical_knowledge.consult_evidence_quality import (
    is_usable_evidence_excerpt,
    protocol_title_for_path,
)
from clinical_knowledge.consult_l2_config import (
    consult_l2_evidence_chunks_per_path,
    consult_l2_evidence_max_chars,
    consult_l2_evidence_max_paths,
)
from clinical_knowledge.protocol_practical_lite import _pick_chunks

GetChunksFn = Callable[[str], list[dict[str, Any]]]

EVIDENCE_BLOCK_IDS = ("diagnosis", "exams", "treatment", "followup")

_CHUNK_TYPES_BY_BLOCK: dict[str, tuple[str, ...]] = {
    "diagnosis": ("diagnostics", "protocol_overview", "diagnosis"),
    "exams": ("diagnostics", "protocol_overview"),
    "treatment": ("treatment", "medications"),
    "followup": ("follow_up", "treatment", "protocol_overview"),
}


def _excerpt_item(
    *,
    block_id: str,
    protocol_path: str,
    section: str,
    excerpt: str,
    page: int | None = None,
    match_status: str = "protocol_excerpt",
    source: str = "rich_chunk",
    chunk_id: str | None = None,
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "protocol_path": protocol_path,
        "section": section,
        "excerpt": excerpt[:2000],
        "page": page,
        "match_status": match_status,
        "source": source,
        "chunk_id": chunk_id,
    }


def _summary_card_excerpts(
    icd_codes: list[str],
    *,
    max_per_block: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {b: [] for b in EVIDENCE_BLOCK_IDS}
    try:
        from clinical_knowledge.protocol_summary.icd_index import find_summary_refs_by_icd
        from clinical_knowledge.protocol_summary.loader import load_summary_by_protocol_id
    except ImportError:
        return out

    seen_refs: set[tuple[str, str]] = set()
    for code in icd_codes[:6]:
        for protocol_id, condition_id in find_summary_refs_by_icd(code, limit=3):
            ref = (protocol_id, condition_id)
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            summary = load_summary_by_protocol_id(protocol_id)
            if summary is None:
                continue
            cond = next(
                (c for c in summary.conditions if c.condition_id == condition_id),
                None,
            )
            if cond is None:
                continue
            src_path = ""
            if summary.source and summary.source.local_path:
                src_path = str(summary.source.local_path)
            elif summary.source and summary.source.url:
                src_path = str(summary.source.url)

            if cond.diagnosis_structure and len(out["diagnosis"]) < max_per_block:
                parts = []
                for comp in (cond.diagnosis_structure.required_components or [])[:3]:
                    parts.append(comp.name)
                if parts:
                    out["diagnosis"].append(
                        _excerpt_item(
                            block_id="diagnosis",
                            protocol_path=src_path or protocol_id,
                            section="Диагноз (summary card)",
                            excerpt="; ".join(parts),
                            match_status="summary_card",
                            source="summary_card",
                        )
                    )

            exams = (cond.required_exams or []) + (cond.conditional_exams or [])
            for ex in exams[:4]:
                if len(out["exams"]) >= max_per_block:
                    break
                txt = ex.name
                if ex.comment:
                    txt = f"{txt}. {ex.comment}"
                if not is_usable_evidence_excerpt(txt):
                    continue
                out["exams"].append(
                    _excerpt_item(
                        block_id="exams",
                        protocol_path=src_path or protocol_id,
                        section="Обследование (summary card)",
                        excerpt=txt,
                        match_status="summary_card",
                        source="summary_card",
                    )
                )

            if cond.treatment and len(out["treatment"]) < max_per_block:
                t_parts: list[str] = []
                for d in (cond.treatment.drugs or [])[:2]:
                    t_parts.append(d.drug_name or d.active_substance or d.drug_group or "")
                for nd in (cond.treatment.non_drug or [])[:2]:
                    t_parts.append(nd.text)
                t_parts = [p for p in t_parts if p and is_usable_evidence_excerpt(p)]
                if t_parts:
                    out["treatment"].append(
                        _excerpt_item(
                            block_id="treatment",
                            protocol_path=src_path or protocol_id,
                            section="Лечение (summary card)",
                            excerpt="; ".join(t_parts),
                            match_status="summary_card",
                            source="summary_card",
                        )
                    )

            for fu in (cond.follow_up or [])[:2]:
                if len(out["followup"]) >= max_per_block:
                    break
                txt = fu.text
                if fu.timing:
                    txt = f"{txt} ({fu.timing})"
                out["followup"].append(
                    _excerpt_item(
                        block_id="followup",
                        protocol_path=src_path or protocol_id,
                        section="Наблюдение (summary card)",
                        excerpt=txt,
                        match_status="summary_card",
                        source="summary_card",
                    )
                )
    return out


def _rich_chunk_excerpts(
    paths: list[str],
    *,
    query: str,
    icd_codes: list[str],
    get_chunks: GetChunksFn,
    max_paths: int,
    chunks_per_path: int,
    max_chars: int,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {b: [] for b in EVIDENCE_BLOCK_IDS}
    total_chars = 0
    for path in paths[:max_paths]:
        p = str(path or "").strip()
        if not p:
            continue
        chunks = get_chunks(p) or []
        if not chunks:
            continue
        try:
            from clinical_knowledge.chunk_tags import chunk_usable_for_retrieval

            chunks = [c for c in chunks if chunk_usable_for_retrieval(c, ambulatory=True)]
        except Exception:
            pass
        for block_id in EVIDENCE_BLOCK_IDS:
            if total_chars >= max_chars:
                break
            picked = _pick_chunks(
                chunks,
                query,
                icd_codes,
                limit=chunks_per_path,
                chunk_types=_CHUNK_TYPES_BY_BLOCK.get(block_id),
            )
            for ch in picked:
                txt = (ch.get("text") or ch.get("lex_text") or "").strip()
                if not is_usable_evidence_excerpt(txt):
                    continue
                if len(txt) < 40:
                    continue
                if total_chars + len(txt) > max_chars:
                    txt = txt[: max(40, max_chars - total_chars)]
                total_chars += len(txt)
                page = ch.get("page_from") or ch.get("page_to")
                out[block_id].append(
                    _excerpt_item(
                        block_id=block_id,
                        protocol_path=p,
                        section=str(ch.get("section_title") or ch.get("kind") or block_id),
                        excerpt=txt,
                        page=int(page) if isinstance(page, (int, float)) else None,
                        match_status="rich_chunk",
                        source="rich_chunk",
                        chunk_id=str(ch.get("chunk_id") or "") or None,
                    )
                )
    return out


def build_evidence_pack(
    *,
    icd_codes: list[str],
    match_paths: list[str],
    get_chunks: GetChunksFn,
    query: str = "",
) -> dict[str, Any]:
    """Секционные выдержки для UI L2."""
    q = (query or "").strip() or " ".join(icd_codes[:6])
    max_paths = consult_l2_evidence_max_paths()
    chunks_per_path = consult_l2_evidence_chunks_per_path()
    max_chars = consult_l2_evidence_max_chars()

    paths = [str(p).strip() for p in match_paths if p][:max_paths]
    summary_blocks = _summary_card_excerpts(icd_codes)
    chunk_blocks = _rich_chunk_excerpts(
        paths,
        query=q,
        icd_codes=icd_codes,
        get_chunks=get_chunks,
        max_paths=max_paths,
        chunks_per_path=chunks_per_path,
        max_chars=max_chars,
    )

    blocks: dict[str, list[dict[str, Any]]] = {}
    fragment_count = 0
    for block_id in EVIDENCE_BLOCK_IDS:
        merged: list[dict[str, Any]] = []
        seen_excerpt: set[str] = set()
        for src in (summary_blocks, chunk_blocks):
            for item in src.get(block_id) or []:
                key = (item.get("protocol_path"), item.get("excerpt", "")[:80])
                if key in seen_excerpt:
                    continue
                seen_excerpt.add(key)
                item = dict(item)
                pth = str(item.get("protocol_path") or "").strip()
                if pth and not item.get("protocol_title"):
                    item["protocol_title"] = protocol_title_for_path(pth)
                merged.append(item)
        blocks[block_id] = merged[:6]
        fragment_count += len(blocks[block_id])

    return {
        "blocks": blocks,
        "fragment_count": fragment_count,
        "paths_used": paths,
        "limits": {
            "max_paths": max_paths,
            "chunks_per_path": chunks_per_path,
            "max_chars": max_chars,
        },
    }


def evidence_pack_to_protocol_rows(
    pack: dict[str, Any],
    *,
    limit_per_path: int = 4,
    max_paths: int = 5,
) -> list[dict[str, Any]]:
    """Компактные строки для UI из evidence pack (без повторного чтения JSONL)."""
    rows: list[dict[str, Any]] = []
    per_path: dict[str, int] = {}
    paths_seen: set[str] = set()
    for _block_id, items in (pack.get("blocks") or {}).items():
        for item in items or []:
            if not isinstance(item, dict):
                continue
            p = str(item.get("protocol_path") or "").strip()
            if not p:
                continue
            if p not in paths_seen:
                if len(paths_seen) >= max_paths:
                    continue
                paths_seen.add(p)
            if per_path.get(p, 0) >= limit_per_path:
                continue
            excerpt = str(item.get("excerpt") or "").strip()
            if len(excerpt) < 40:
                continue
            page = item.get("page")
            rows.append(
                {
                    "path": p,
                    "text": excerpt,
                    "excerpt": excerpt[:2000],
                    "kind": str(item.get("block_id") or item.get("source") or "fragment"),
                    "section_title": str(item.get("section") or ""),
                    "page_from": int(page) if isinstance(page, (int, float)) else None,
                }
            )
            per_path[p] = per_path.get(p, 0) + 1
    return rows
