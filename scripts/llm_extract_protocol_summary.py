#!/usr/bin/env python3
"""Extract one protocol summary via LLM or structured fallback."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.protocol_summary.builder import publish_summaries  # noqa: E402
from clinical_knowledge.protocol_summary.llm_extractor import extract_protocol_summary_llm  # noqa: E402
from clinical_knowledge.protocol_summary.loader import clear_protocol_summary_cache, export_summary_json  # noqa: E402
from clinical_knowledge.protocol_summary.source_text import (  # noqa: E402
    build_source_text_document,
    load_source_text,
    save_source_text,
)
from clinical_knowledge.protocol_summary.validator import validate_protocol_summary, write_validation_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-id", type=str, required=True)
    ap.add_argument("--path", type=str, default=None, help="catalog path if source_text missing")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    doc = load_source_text(args.protocol_id)
    if doc is None:
        if not args.path:
            print("source_text missing; pass --path", file=sys.stderr)
            return 1
        doc = build_source_text_document(args.path.replace("\\", "/"))
        save_source_text(doc)

    summary = extract_protocol_summary_llm(doc, use_llm=not args.no_llm)
    blob = json.dumps(doc.get("sections") or {}, ensure_ascii=False)
    result = validate_protocol_summary(summary, source_blob=blob)
    summary.validation = result
    write_validation_report(summary, result)

    drafts = ROOT / "data" / "protocol_summaries" / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        (drafts / f"{summary.protocol_id}.yaml").write_text(
            yaml.safe_dump(summary.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    except ImportError:
        pass
    export_summary_json(summary)
    if args.publish:
        publish_summaries([summary])
        clear_protocol_summary_cache()
    print(f"{summary.protocol_id}: {result.status} extractor={summary.extraction_metadata.extractor}")
    return 0 if result.status != "invalid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
