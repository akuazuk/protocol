#!/usr/bin/env python3
"""LLM-enrich для непокрытых PDF всего каталога (CORPUS_LLM_ENRICH=1, GEMINI_API_KEY)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CORPUS_LLM_ENRICH", "1")
ENRICH_DIR = ROOT / "data" / "gastro_mvp" / "enrichment"


def main() -> int:
    from clinical_knowledge.coverage import load_rules_coverage_report
    from clinical_knowledge.enrichment_samples import sample_text_for_pdf
    from clinical_knowledge.llm_enrich import enrich_condition_text
    from clinical_knowledge.rules_from_path import infer_path_condition

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

    coverage = load_rules_coverage_report()
    without = list(coverage.get("without_rules") or [])
    if not without:
        print(json.dumps({"message": "all_pdfs_covered", "without_rules": 0}, indent=2))
        return 0

    results: dict[str, str] = {}
    for sp in without:
        inferred = infer_path_condition(sp)
        cid = inferred[0] if inferred else f"pdf_{Path(sp).stem[:24]}"
        sample = sample_text_for_pdf(sp, chunks)
        if not sample:
            results[Path(sp).name[:50]] = "no_sample"
            continue
        out = enrich_condition_text(
            cid,
            sample,
            model=model,
            generate_fn=generate_gemini,
            extract_text_fn=_extract_gemini_text,
        )
        if out and isinstance(out, dict):
            out["source_path"] = sp.replace("\\", "/")
            cache_path = ENRICH_DIR / f"{cid}_{out.get('text_hash', 'x')}.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        results[Path(sp).name[:50]] = "ok" if out else "failed"

    summary_path = ROOT / "data" / "gastro_mvp" / "catalog_enrichment_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "without_rules_input": len(without),
                "processed": len(results),
                "ok": sum(1 for v in results.values() if v == "ok"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
