#!/usr/bin/env python3
"""Извлечь path/corpus-правила по всем PDF каталога (все рубрики Минздрава РБ)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from clinical_knowledge.catalog_build import build_catalog_rules

    summary = build_catalog_rules()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("pdfs_total") else 1


if __name__ == "__main__":
    raise SystemExit(main())
