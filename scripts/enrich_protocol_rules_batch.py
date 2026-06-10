#!/usr/bin/env python3
"""Batch LLM-enrichment правил протоколов (фаза 5): кэш в data/catalog/enrichment/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_load import load_project_env

load_project_env(ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-enrichment правил для PDF без rules")
    parser.add_argument("--limit", type=int, default=0, help="Макс. PDF за запуск (0 = все)")
    parser.add_argument("--force", action="store_true", help="Пересобрать даже при наличии кэша")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os_enrich = __import__("os").environ
    if not args.dry_run:
        os_enrich["CORPUS_LLM_ENRICH"] = "1"

    from clinical_knowledge.catalog_full_build import build_catalog_full

    progress_log: list[dict] = []

    def on_progress(ev: dict) -> None:
        progress_log.append(ev)
        if ev.get("stage") in ("enrich", "rules", "done"):
            print(f"[{ev.get('pct', 0):3d}%] {ev.get('label_ru', '')}")

    if args.force:
        enrich_dir = ROOT / "data" / "catalog" / "enrichment"
        if enrich_dir.is_dir():
            for p in enrich_dir.glob("*.json"):
                p.unlink()

    summary = build_catalog_full(
        use_llm=not args.dry_run,
        llm_limit=args.limit,
        on_progress=on_progress,
    )
    out_path = ROOT / "output" / "enrich_batch_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
