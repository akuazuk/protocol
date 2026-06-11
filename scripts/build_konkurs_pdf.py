#!/usr/bin/env python3
"""Сборка PDF-пакета конкурса Белинфонд (основной формат подачи).

Генерирует print-ready HTML + PDF через Chrome headless.
Дополнительно обновляет графики и (опционально) docx.

  python3 scripts/build_konkurs_pdf.py
  python3 scripts/build_konkurs_pdf.py --html-only
  python3 scripts/build_konkurs_pdf.py --with-docx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from build_architecture_pdf import html_to_pdf, pdf_path_for_html  # noqa: E402
from konkurs_docx_helpers import generate_charts  # noqa: E402
from konkurs_html import KONKURS_DIR, write_all_html  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = KONKURS_DIR / "_assets"


def build_pdfs(html_files: list[Path]) -> None:
    for html_path in html_files:
        pdf_path = pdf_path_for_html(html_path)
        html_to_pdf(html_path, pdf_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Конкурс Белинфонд: HTML + PDF")
    parser.add_argument("--html-only", action="store_true", help="Только HTML, без PDF")
    parser.add_argument("--with-docx", action="store_true", help="Также пересобрать docx")
    args = parser.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    print("Charts...")
    generate_charts(ASSETS)

    print("HTML...")
    html_files = write_all_html(KONKURS_DIR)
    for p in html_files:
        print(f"  {p.relative_to(ROOT)}")

    if not args.html_only:
        print("PDF...")
        build_pdfs(html_files)

    if args.with_docx:
        from fill_konkurs_docx import main as fill_docx  # noqa: E402

        print("DOCX (legacy)...")
        fill_docx()

    print("OK:", KONKURS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
