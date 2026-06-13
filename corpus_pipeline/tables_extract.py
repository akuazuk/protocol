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
    """Первая строка - заголовок; пустые заголовки заменяются на Столбец N."""
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


def _columns_signature(columns: list[str]) -> tuple[str, ...]:
    """Нормализованная подпись заголовка для сравнения таблиц между страницами."""
    return tuple((c or "").strip().lower() for c in columns)


def _is_synthetic_header(columns: list[str]) -> bool:
    return all((c or "").strip().lower().startswith("столбец") for c in columns) if columns else True


def merge_multipage_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Склеивает таблицы, продолжающиеся через границу страниц.

    Эвристика: таблица на следующей странице (или той же при последовательных блоках)
    объединяется с предыдущей, если совпадает заголовок столбцов либо у продолжения
    синтетический заголовок и совпадает число столбцов. Объединённая таблица сохраняет
    исходный заголовок и расширяет диапазон page_from..page_to.
    """
    if not tables:
        return tables

    ordered = sorted(
        tables,
        key=lambda t: (int(t.get("page") or 1), int(t.get("table_index_on_page") or 0)),
    )

    merged: list[dict[str, Any]] = []
    for t in ordered:
        page = int(t.get("page") or 1)
        cols = list(t.get("columns") or [])
        rows = list(t.get("rows") or [])
        width = len(cols)
        t.setdefault("page_from", page)
        t.setdefault("page_to", page)

        if merged:
            prev = merged[-1]
            prev_cols = list(prev.get("columns") or [])
            same_header = _columns_signature(prev_cols) == _columns_signature(cols)
            continuation = (
                len(prev_cols) == width
                and width > 0
                and (same_header or _is_synthetic_header(cols))
            )
            page_adjacent = page <= int(prev.get("page_to") or prev.get("page") or page) + 1
            if continuation and page_adjacent:
                prev_rows = list(prev.get("rows") or [])
                # При совпадении заголовка строки продолжения - это данные, не повтор шапки.
                prev_rows.extend(rows)
                prev["rows"] = prev_rows
                prev["page_to"] = max(int(prev.get("page_to") or page), page)
                prev["raw_markdown"] = table_to_markdown(prev_cols, prev_rows)
                prev["normalized"] = {
                    "columns": prev_cols,
                    "row_count": len(prev_rows),
                }
                continue

        merged.append(t)

    return merged


# Обратная совместимость
def _table_to_markdown(header: list[str], rows: list[list[str]]) -> str:
    return table_to_markdown(header, rows)
