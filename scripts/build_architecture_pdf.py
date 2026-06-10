#!/usr/bin/env python3
"""Сборка PDF из print-ready HTML (Chrome headless).

Пример:
  python3 scripts/build_architecture_pdf.py
  python3 scripts/build_architecture_pdf.py docs/architecture-stages-print.html
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HTML = ROOT / "docs" / "architecture-kravira-fhir-mis-print.html"


def _chrome_bin() -> str | None:
    for cand in (
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ):
        if cand and Path(cand).is_file():
            return cand
    return None


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = _chrome_bin()
    if not chrome:
        raise SystemExit(
            "Не найден Chrome/Chromium для headless PDF. "
            "Установите Chrome или сохраните PDF вручную: открыть HTML → Печать → PDF."
        )
    html_path = html_path.resolve()
    pdf_path = pdf_path.resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    url = html_path.as_uri()
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or not pdf_path.is_file():
        err = (proc.stderr or proc.stdout or "").strip()[:500]
        raise SystemExit(f"Chrome не создал PDF ({proc.returncode}): {err}")
    print(f"OK: {pdf_path} ({pdf_path.stat().st_size // 1024} KiB)")


def main() -> int:
    parser = argparse.ArgumentParser(description="HTML print → PDF через Chrome headless")
    parser.add_argument(
        "html",
        nargs="?",
        default=str(DEFAULT_HTML),
        help="Путь к HTML (по умолчанию architecture-kravira-fhir-mis-print.html)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="",
        help="PDF (по умолчанию: рядом с HTML, имя .pdf)",
    )
    args = parser.parse_args()
    html_path = Path(args.html)
    if not html_path.is_file():
        raise SystemExit(f"Нет файла: {html_path}")
    if args.output:
        pdf_path = Path(args.output)
    else:
        pdf_path = html_path.with_suffix(".pdf")
        if pdf_path.name.endswith("-print.pdf"):
            pdf_path = pdf_path.with_name(pdf_path.name.replace("-print.pdf", ".pdf"))
    html_to_pdf(html_path, pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
