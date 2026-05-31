#!/usr/bin/env python3
"""LLM-обогащение нозологий гастро (CORPUS_LLM_ENRICH=1, нужен GEMINI_API_KEY)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CORPUS_LLM_ENRICH", "1")


def main() -> int:
    from clinical_knowledge.enrichment_samples import sample_text_for_condition
    from clinical_knowledge.llm_enrich import enrich_condition_text
    from clinical_knowledge.loader import load_conditions

    chunks = ROOT / "output" / "chunks" / "chunks.jsonl"
    if not chunks.is_file():
        print("WARN: нет output/chunks/chunks.jsonl")
        return 1

    try:
        from rag_server import _extract_gemini_text, generate_gemini, get_gemini

        model = get_gemini()
    except Exception as e:
        print(f"SKIP: Gemini недоступен ({e})")
        return 0

    results: dict[str, str] = {}
    for cid in sorted(load_conditions().keys()):
        sample = sample_text_for_condition(cid, chunks)
        if not sample:
            results[cid] = "no_sample"
            continue
        out = enrich_condition_text(
            cid,
            sample,
            model=model,
            generate_fn=generate_gemini,
            extract_text_fn=_extract_gemini_text,
        )
        results[cid] = "ok" if out else "failed"

    summary_path = ROOT / "data" / "gastro_mvp" / "enrichment_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
