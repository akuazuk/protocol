"""Извлечение таблиц из PDF (pdfplumber)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def table_to_markdown(header: list[str], rows: list[list[str]]) -> str:
    """Markdown-таблица для индексации и отображения; header может быть синтетическим."""
    if not header and not rows:
        return ""
    if not header:
        max_w = max((len(r) for r in rows), default=0)
        header = [f"Столбец {j + 1}" for j in range(max_w)]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        pad = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(pad[: len(header)]) + " |")
    return "\n".join(lines)


def _pad_row(row: list[str], width: int) -> list[str]:
    row = row + [""] * (width - len(row))
    return row[:width]


def _normalize_pdf_table(table: list[list[Any]]) -> tuple[list[str], list[list[str]]] | None:
    """Первая строка — заголовок; пустые заголовки заменяются на Столбец N."""
    if not table:
        return None
    cells = [[str(c or "").strip() for c in row] for row in table]
    cells = [r for r in cells if any(x for x in r)]
    if len(cells) < 2:
        return None
    max_w = max(len(r) for r in cells)
    header = _pad_row(cells[0], max_w)
    data_rows = [_pad_row(r, max_w) for r in cells[1:]]
    if not any(header):
        header = [f"Столбец {j + 1}" for j in range(max_w)]
    return header, data_rows


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
                    norm = _normalize_pdf_table(table)
                    if not norm:
                        continue
                    header, rows = norm
                    if not rows:
                        continue
                    raw_md = table_to_markdown(header, rows)
                    if len(raw_md.strip()) < 30:
                        continue
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
                            "extraction_confidence": float(
                                os.environ.get("CORPUS_TABLE_CONFIDENCE", "0.75")
                            ),
                        }
                    )
    except Exception:
        return out
    return out


def merge_multipage_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Заглушка объединения; полная эвристика — в следующих версиях."""
    for t in tables:
        t.setdefault("page_from", t.get("page"))
        t.setdefault("page_to", t.get("page"))
    return tables


# Обратная совместимость
def _table_to_markdown(header: list[str], rows: list[list[str]]) -> str:
    return table_to_markdown(header, rows)
