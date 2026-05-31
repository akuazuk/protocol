#!/usr/bin/env python3
"""Export summary RAG chunks jsonl."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.protocol_summary.loader import load_protocol_summaries  # noqa: E402
from clinical_knowledge.protocol_summary.summary_to_rag import write_summary_rag_jsonl  # noqa: E402


def main() -> int:
    path = write_summary_rag_jsonl(list(load_protocol_summaries(usable_only=False)))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
