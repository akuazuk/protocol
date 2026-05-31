"""Сбор текстовых сэмплов по нозологиям из chunks.jsonl для LLM-enrich."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .rules_from_corpus import (
    CONDITION_DIAG_PATTERNS,
    IBD_NUMBERED_DIAG_SECTIONS,
    _collapse_ws,
    load_chunks_exact,
)

ROOT = Path(__file__).resolve().parent.parent

CONDITION_PDF_HINTS: dict[str, list[str]] = {
    "gerd": ["пищевода_желудка_двенадцатиперстной", "пищевода_желудка"],
    "gastritis": ["гастрит"],
    "peptic_ulcer": ["пищевода_желудка"],
    "ulcerative_colitis": ["язвенным_колитом", "кишечника"],
    "crohn": ["болезнью_крона", "крона"],
    "celiac": ["целиак"],
    "functional_dyspepsia": ["пищевода_желудка", "диспепс"],
    "acute_pancreatitis": ["панкреат"],
    "acute_appendicitis": ["аппендицит"],
}


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
