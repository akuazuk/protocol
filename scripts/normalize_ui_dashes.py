#!/usr/bin/env python3
"""Замена длинного/среднего тире на короткий дефис в UI-текстах проекта."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Только пользовательские тексты приложения и связанных материалов (не extracted corpus).
TARGETS = [
    ROOT / "index.html",
    ROOT / "rag_server.py",
    ROOT / "README.md",
    ROOT / "docs/mvp-presentation.html",
    ROOT / "docs/ministry-brief-print.html",
    ROOT / "docs/ministry-brief-ru.md",
    ROOT / "docs/deployment-belarus.md",
    ROOT / "docs/roadmap-mis.md",
]


def normalize_dashes(text: str) -> str:
    """Заменить em/en dash, не трогая отступы и прочие пробелы."""
    text = text.replace("\u2013", "-")
    return re.sub(r"\s*\u2014\s*", " - ", text)


def main() -> int:
    changed = 0
    for path in TARGETS:
        if not path.is_file():
            print(f"skip (missing): {path.relative_to(ROOT)}")
            continue
        raw = path.read_text(encoding="utf-8")
        norm = normalize_dashes(raw)
        if norm != raw:
            path.write_text(norm, encoding="utf-8")
            n = raw.count("\u2014") + raw.count("\u2013")
            print(f"updated {path.relative_to(ROOT)} ({n} dash chars)")
            changed += 1
        else:
            print(f"ok {path.relative_to(ROOT)}")
    print(f"done: {changed} file(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
