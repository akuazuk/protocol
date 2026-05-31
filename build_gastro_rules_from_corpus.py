#!/usr/bin/env python3
"""Извлечь правила гастро из chunks.jsonl (КП №185 пищевод/желудок/ДПК) и записать в data/gastro_mvp/rules/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_knowledge.rules_from_corpus import (
    extract_rules_from_chunks,
    load_chunks_for_source,
    merge_rules_into_gastro_mvp,
)

KP185_SUBSTR = "пищевода_желудка_двенадцатиперстной"
LOGICAL_DOC = "_L8"


def main() -> None:
    chunks_path = ROOT / "output" / "chunks" / "chunks.jsonl"
    chunks = load_chunks_for_source(chunks_path, KP185_SUBSTR, logical_suffix=LOGICAL_DOC)
    if not chunks:
        print(f"WARN: нет чанков для {KP185_SUBSTR}{LOGICAL_DOC}")
        sys.exit(1)

    extracted = extract_rules_from_chunks(
        chunks,
        protocol_id="gastro_esophagus_stomach_adult_2025_185",
    )
    out_dir = ROOT / "data" / "gastro_mvp" / "rules"
    counts = merge_rules_into_gastro_mvp(extracted, out_dir)

    summary = {
        "source_pdf_substr": KP185_SUBSTR,
        "logical_doc": LOGICAL_DOC,
        "chunks_scanned": len(chunks),
        "rules_extracted": counts,
    }
    (ROOT / "data" / "gastro_mvp" / "rules_extraction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
