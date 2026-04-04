#!/usr/bin/env python3
"""
Извлекает текст из PDF протоколов для полнотекстового и семантического поиска.
Требует: pip install pymupdf
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "minzdrav_protocols"
OUT = ROOT / "corpus.json"
MAX_CHARS = 8000

_WS = re.compile(r"\s+")


def extract_text(pdf_path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Установите: pip install pymupdf", file=sys.stderr)
        raise SystemExit(1)

    doc = fitz.open(pdf_path)
    parts: list[str] = []
    for page in doc:
        parts.append(page.get_text("text") or "")
    doc.close()
    text = _WS.sub(" ", "\n".join(parts)).strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "…"
    return text


def main() -> None:
    if not DATA.is_dir():
        raise SystemExit(f"Нет каталога: {DATA}")

    pdfs = sorted(DATA.rglob("*.pdf"))
    rows: list[dict] = []
    for p in pdfs:
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        try:
            text = extract_text(p)
        except Exception as e:
            print(f"Предупреждение {rel}: {e}", file=sys.stderr)
            text = ""
        rows.append({"path": rel, "text": text})

    OUT.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    n_nonempty = sum(1 for r in rows if r["text"])
    print(f"Корпус: {len(rows)} файлов, с текстом: {n_nonempty} → {OUT}")


if __name__ == "__main__":
    main()
