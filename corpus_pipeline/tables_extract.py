"""Извлечение таблиц из PDF (pdfplumber)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def extract_tables_from_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    try:
        import pdfplumber
    except ImportError:
        return []

    out: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pi, page in enumerate(pdf.pages):
                tables = page.extract_tables() or []
                for ti, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                    header = [str(c or "").strip() for c in table[0]]
                    rows = []
                    for row in table[1:]:
                        cells = [str(c or "").strip() for c in row]
                        if any(cells):
                            rows.append(cells)
                    raw_md = _table_to_markdown(header, rows)
                    out.append(
                        {
                            "page": pi + 1,
                            "table_index_on_page": ti,
                            "title": None,
                            "columns": header,
                            "rows": rows,
                            "raw_markdown": raw_md,
                            "normalized": {
                                "columns": header,
                                "row_count": len(rows),
                            },
                            "extraction_confidence": 0.7,
                        }
                    )
    except Exception:
        return out
    return out


def _table_to_markdown(header: list[str], rows: list[list[str]]) -> str:
    if not header:
        return ""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        pad = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(pad[: len(header)]) + " |")
    return "\n".join(lines)


def merge_multipage_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Заглушка объединения; полная эвристика — в следующих версиях."""
    for t in tables:
        t.setdefault("page_from", t.get("page"))
        t.setdefault("page_to", t.get("page"))
    return tables
