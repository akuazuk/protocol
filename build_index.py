#!/usr/bin/env python3
"""Строит index.csv по дереву minzdrav_protocols/."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "minzdrav_protocols"
OUT = ROOT / "index.csv"
OUT_JSON = ROOT / "protocols.json"

YEAR_RE = re.compile(r"(20\d{2}|19\d{2})")


def iso_mtime(p: Path) -> str:
    ts = p.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def guess_years(name: str) -> str:
    years = sorted(set(YEAR_RE.findall(name)), reverse=True)
    return ";".join(years) if years else ""


def main() -> None:
    if not DATA.is_dir():
        raise SystemExit(f"Нет каталога: {DATA}")

    pdfs = sorted(DATA.rglob("*.pdf"))
    basenames = [p.name for p in pdfs]
    dup_count = Counter(basenames)

    json_rows: list[dict] = []
    for p in pdfs:
        rel = p.relative_to(ROOT)
        name = p.name
        title = re.sub(r"\.pdf$", "", name, flags=re.I)
        json_rows.append(
            {
                "path": str(rel).replace("\\", "/"),
                "category": p.parent.name,
                "filename": name,
                "title": title,
            }
        )
    OUT_JSON.write_text(
        json.dumps(json_rows, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "relative_path",
                "category",
                "filename",
                "size_bytes",
                "modified_utc",
                "years_in_filename",
                "has_post_mz",
                "same_basename_count",
            ]
        )
        for p in pdfs:
            rel = p.relative_to(ROOT)
            name = p.name
            w.writerow(
                [
                    str(rel).replace("\\", "/"),
                    p.parent.name,
                    name,
                    p.stat().st_size,
                    iso_mtime(p),
                    guess_years(name),
                    "yes" if ("пост_МЗ" in name or "постановление_МЗ" in name) else "no",
                    dup_count[name],
                ]
            )

    print(f"Записано строк: {len(pdfs)} → {OUT}")
    print(f"JSON для поиска: {OUT_JSON}")


if __name__ == "__main__":
    main()
