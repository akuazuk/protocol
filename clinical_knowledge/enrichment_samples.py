"""Сбор текстовых сэмплов по нозологиям из chunks.jsonl для LLM-enrich."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .condition_registry import CONDITION_BY_ID
from .rules_from_corpus import (
    CONDITION_DIAG_PATTERNS,
    IBD_NUMBERED_DIAG_SECTIONS,
    _collapse_ws,
    load_chunks_exact,
)

ROOT = Path(__file__).resolve().parent.parent

CONDITION_PDF_HINTS: dict[str, list[str]] = {
    cid: list(c.path_hints or c.card_keywords[:2])
    for cid, c in CONDITION_BY_ID.items()
    if c.path_hints or c.card_keywords
}


def sample_text_for_pdf(
    source_path: str,
    chunks_path: Path,
    *,
    max_chars: int = 10_000,
) -> str:
    """Текстовый сэмпл из PDF для LLM-enrich (приоритет — блоки диагноза/классификации)."""
    from .rules_from_corpus import _collapse_ws, load_chunks_exact, pick_best_logical_chunks

    chunks = pick_best_logical_chunks(load_chunks_exact(chunks_path, source_path))
    if not chunks:
        return ""
    scored: list[tuple[int, str]] = []
    for c in chunks:
        t = c.get("text") or ""
        if len(t) < 60:
            continue
        low = t.lower()
        score = sum(
            1
            for kw in (
                "диагноз",
                "классиф",
                "критери",
                "формулиров",
                "диагностик",
                "лечени",
            )
            if kw in low
        )
        if score:
            scored.append((score, t[:3000]))
    scored.sort(key=lambda x: -x[0])
    parts = [t for _, t in scored[:6]]
    if not parts:
        blob = _collapse_ws(" ".join(c.get("text") or "" for c in chunks))
        return blob[:max_chars]
    return "\n\n".join(parts)[:max_chars]


def sample_text_for_condition(
    condition_id: str,
    chunks_path: Path,
    *,
    max_chars: int = 10_000,
) -> str:
    hints = CONDITION_PDF_HINTS.get(condition_id) or []
    parts: list[str] = []
    registry = ROOT / "data" / "gastro_mvp" / "protocol_registry.jsonl"
    paths: list[str] = []
    if registry.is_file():
        for line in registry.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sp = (row.get("source_path") or "").replace("\\", "/")
            if any(h in sp.lower() for h in hints):
                if sp not in paths:
                    paths.append(sp)
    for sp in paths[:3]:
        chunks = load_chunks_exact(chunks_path, sp)
        blob = _collapse_ws(" ".join((c.get("text") or "") for c in chunks))
        sections = IBD_NUMBERED_DIAG_SECTIONS.get(condition_id)
        if sections and any(s in blob for s in sections[:2]):
            parts.append(blob[:max_chars])
            break
        for c in chunks:
            t = c.get("text") or ""
            if len(t) < 80:
                continue
            for _, pat in CONDITION_DIAG_PATTERNS:
                if pat.search(t):
                    parts.append(t[:2500])
                    break
            if sum(len(p) for p in parts) >= max_chars:
                break
        if sum(len(p) for p in parts) >= max_chars:
            break
    return "\n\n".join(parts)[:max_chars]
