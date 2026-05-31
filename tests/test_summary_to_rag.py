"""Tests for summary RAG chunks."""
from __future__ import annotations

from pathlib import Path

import yaml

from clinical_knowledge.protocol_summary.schema import ProtocolSummary
from clinical_knowledge.protocol_summary.summary_to_rag import summary_to_rag_chunks

FIX = Path(__file__).resolve().parent / "fixtures" / "protocol_summaries" / "yaml"


def test_summary_chunks_created():
    data = yaml.safe_load((FIX / "test_gastro_k30.yaml").read_text(encoding="utf-8"))
    summary = ProtocolSummary.model_validate(data)
    chunks = summary_to_rag_chunks(summary)
    assert chunks
    assert all(c.get("generated_from_summary") for c in chunks)
    types = {c["section_type"] for c in chunks}
    assert "summary_overview" in types
    assert "summary_red_flags" in types
