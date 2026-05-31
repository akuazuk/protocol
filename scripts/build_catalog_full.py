#!/usr/bin/env python3
"""Полная структуризация каталога (478 PDF): conditions + rules как gastro MVP."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="LLM-enrich для PDF без правил (GEMINI_API_KEY)")
    parser.add_argument("--llm-limit", type=int, default=0, help="Макс. PDF для LLM (0 = без лимита)")
    parser.add_argument("--quiet", action="store_true", help="Только финальный JSON")
    args = parser.parse_args()

    from clinical_knowledge.catalog_full_build import build_catalog_full
    from clinical_knowledge.loader import clear_clinical_knowledge_cache

    def on_progress(ev: dict) -> None:
        if args.quiet:
            return
        print(json.dumps(ev, ensure_ascii=False), flush=True)

    summary = build_catalog_full(
        on_progress=on_progress,
        use_llm=args.llm,
        llm_limit=args.llm_limit,
    )
    clear_clinical_knowledge_cache()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("pdfs_total") else 1


if __name__ == "__main__":
    raise SystemExit(main())
