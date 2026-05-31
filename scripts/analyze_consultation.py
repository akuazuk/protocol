#!/usr/bin/env python3
"""CLI: структурный анализ консультативного заключения (ТЗ раздел 5).

Примеры:
    python -m scripts.analyze_consultation --file path/to/kz.pdf
    python -m scripts.analyze_consultation --file kz.txt --markdown out.md
    python -m scripts.analyze_consultation --folder data/examples/consultations --output reports/

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

from clinical_knowledge.consult_analysis import analyze_consultation_text  # noqa: E402


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md", ".json"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print("pypdf не установлен — не могу прочитать PDF.", file=sys.stderr)
            return ""
        try:
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка чтения PDF {path.name}: {exc}", file=sys.stderr)
            return ""
    print(f"Неподдерживаемый тип файла: {path.name}", file=sys.stderr)
    return ""


def _analyze_file(path: Path, *, markdown_out: Path | None, out_dir: Path | None) -> dict:
    text = _extract_text(path)
    res = analyze_consultation_text(
        text, consultation_id=path.stem, source_file=path.name,
        source_file_type=path.suffix.lstrip("."), with_markdown=True,
    )
    md = res.pop("report_markdown", "")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{path.stem}.report.json").write_text(
            json.dumps(res["compliance"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / f"{path.stem}.report.md").write_text(md, encoding="utf-8")
    if markdown_out:
        markdown_out.write_text(md, encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Структурный анализ КЗ по клиническим протоколам")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=str, help="путь к одному КЗ (PDF/TXT/JSON)")
    g.add_argument("--folder", type=str, help="папка с КЗ для batch-анализа")
    ap.add_argument("--markdown", type=str, default=None, help="куда сохранить MD-отчёт (для --file)")
    ap.add_argument("--output", type=str, default=None, help="папка для JSON/MD отчётов")
    ap.add_argument("--quiet", action="store_true", help="не печатать JSON в stdout")
    args = ap.parse_args()

    out_dir = Path(args.output) if args.output else None

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Файл не найден: {path}", file=sys.stderr)
            return 2
        res = _analyze_file(
            path, markdown_out=Path(args.markdown) if args.markdown else None, out_dir=out_dir
        )
        if not args.quiet:
            print(json.dumps(res["compliance"], ensure_ascii=False, indent=2))
        return 0

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Папка не найдена: {folder}", file=sys.stderr)
        return 2
    files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in (".pdf", ".txt", ".json", ".md")
    )
    summary = []
    for p in files:
        try:
            res = _analyze_file(p, markdown_out=None, out_dir=out_dir)
            comp = res["compliance"]
            summary.append({
                "file": p.name,
                "overall_status": comp.get("overall_status"),
                "overall_score": comp.get("score_breakdown", {}).get("overall_score"),
            })
        except Exception as exc:  # batch не должен падать на одном файле (ТЗ 4.6)
            summary.append({"file": p.name, "error": str(exc)[:200]})
    if not args.quiet:
        print(json.dumps({"analyzed": len(summary), "results": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
