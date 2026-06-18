"""Извлечение правил из table_block чанков корпуса (ТЗ improve_kz §11)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .consult_schema import SourceRef
from .rule_model import ProtocolRule

ROOT = Path(__file__).resolve().parent.parent


def resolve_table_chunks_path(path: Path | None = None) -> Path:
    from .catalog_build import resolve_chunks_path

    return path or resolve_chunks_path()

_EXAM_HEADERS = re.compile(
    r"обязательн|рекомендован|диагност|лабор|инструмент",
    re.I,
)
_DRUG_ROW = re.compile(r"(\d+(?:[.,]\d+)?)\s*(мг|мкг|г|мл|мг/кг)", re.I)


def _classify_table(headers: list[str], rows: list[list[str]]) -> str:
    blob = " ".join(headers + [c for row in rows for c in row]).lower()
    if "доз" in blob or _DRUG_ROW.search(blob):
        return "drug_dose_rule"
    if "обязательн" in blob or "минималь" in blob:
        return "required_exam_rule"
    if "дополнительн" in blob or "показан" in blob:
        return "conditional_exam_rule"
    if "контрол" in blob or "повтор" in blob:
        return "follow_up_rule"
    if _EXAM_HEADERS.search(blob):
        return "required_exam_rule"
    return "informational_rule"


def _items_from_rows(rows: list[list[str]]) -> list[str]:
    out: list[str] = []
    for row in rows:
        for cell in row:
            t = (cell or "").strip()
            if len(t) > 2 and not t.isdigit():
                out.append(t[:200])
    return out[:30]


def rule_from_table_chunk(chunk: dict[str, Any]) -> ProtocolRule | None:
    """Один table_block чанк → ProtocolRule или None."""
    rows = chunk.get("rows") or chunk.get("table_rows")
    if not rows and chunk.get("text"):
        # fallback: строки через перенос
        lines = [ln.strip(" \t-\u2013\u2014") for ln in str(chunk["text"]).splitlines() if ln.strip()]
        rows = [[ln] for ln in lines if len(ln) > 3]
    if not rows:
        return None
    headers = list(chunk.get("columns") or chunk.get("header") or [])
    if isinstance(headers, str):
        headers = [headers]
    rule_type = _classify_table(headers, rows)  # type: ignore[arg-type]
    items = _items_from_rows(rows)  # type: ignore[arg-type]
    if not items:
        return None
    src_path = str(chunk.get("source_path") or "")
    page = chunk.get("page") or chunk.get("page_start")
    return ProtocolRule(
        rule_id=f"tbl_{chunk.get('chunk_id') or chunk.get('table_id') or 'x'}",
        protocol_id=str(chunk.get("doc_id") or chunk.get("pdf_doc_id") or ""),
        rule_type=rule_type,  # type: ignore[arg-type]
        severity="required" if rule_type == "required_exam_rule" else "recommended",
        evidence_targets=["performed_exams", "recommended_exams"]
        if "exam" in rule_type
        else ["medications", "treatment"],
        expected_items=items,
        source=SourceRef(
            local_path=src_path or None,
            protocol_id=str(chunk.get("doc_id") or "") or None,
            page_start=int(page) if isinstance(page, int) else None,
            section_title=str(chunk.get("section_title") or chunk.get("section_path") or ""),
            quote=(chunk.get("text") or "")[:400] or None,
        ),
        confidence=0.75,
    )


def extract_rules_from_chunks_path(path: Path | None = None, *, limit: int = 500) -> list[ProtocolRule]:
    """Читает rich/legacy chunks JSONL и извлекает правила из table_block."""
    p = resolve_table_chunks_path(path)
    if not p.is_file():
        return []
    rules: list[ProtocolRule] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            if len(rules) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                ch = json.loads(line)
            except json.JSONDecodeError:
                continue
            ctype = str(ch.get("chunk_type") or "")
            if ctype not in ("table_block", "table", "drug_list"):
                continue
            rule = rule_from_table_chunk(ch)
            if rule:
                rules.append(rule)
    return rules
