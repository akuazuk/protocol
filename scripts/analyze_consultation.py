#!/usr/bin/env python3
"""CLI: структурный анализ консультативного заключения (ТЗ раздел 5).

Примеры (эквивалент check-kz из ТЗ):
    python -m scripts.analyze_consultation --file path/to/kz.pdf
    python -m scripts.analyze_consultation --file kz.txt --markdown out.md
    python -m scripts.analyze_consultation --folder data/examples/consultations --output data/reports/kz_checks

Извлечение текста: TXT — как есть; PDF — через pypdf (текстовый слой, без OCR).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.batch_runner import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    analyze_file,
    run_batch,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Проверка КЗ на соответствие требованиям РБ и клиническим протоколам (check-kz)",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=str, help="путь к одному КЗ (PDF/TXT/JSON)")
    g.add_argument("--folder", type=str, help="папка с КЗ для batch-анализа")
    ap.add_argument("--markdown", type=str, default=None, help="куда сохранить MD-отчёт (для --file)")
    ap.add_argument(
        "--output", type=str, default=None,
        help=f"папка для JSON/MD отчётов (по умолчанию для batch: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument("--quiet", action="store_true", help="не печатать JSON в stdout")
    args = ap.parse_args()

    out_dir = Path(args.output) if args.output else None

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Файл не найден: {path}", file=sys.stderr)
            return 2
        res = analyze_file(path, out_dir=out_dir)
        md = res.pop("report_markdown", "")
        if args.markdown:
            Path(args.markdown).write_text(md, encoding="utf-8")
        if not args.quiet:
            print(json.dumps(res["compliance"], ensure_ascii=False, indent=2))
        return 0

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Папка не найдена: {folder}", file=sys.stderr)
        return 2
    batch_out = out_dir or DEFAULT_OUTPUT_DIR
    summary = run_batch(folder, out_dir=batch_out)
    if not args.quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
