"""Телеметрия поиска: все пути (assist, funnel, icd-fast)."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.search_telemetry import (
    log_protocol_search_from_payload,
    iter_protocol_search_events,
)


def test_log_from_payload_icd_fast(tmp_path: Path, monkeypatch):
    fb = tmp_path / "feedback"
    fb.mkdir()
    monkeypatch.setenv("ML_FEEDBACK_DIR", str(fb))
    log_protocol_search_from_payload(
        query="кашель J06.9",
        payload={
            "llm_json": {"protocols": [{"path": "a.pdf", "confidence_score": 92}]},
            "retrieved_count": 3,
        },
        icd_codes=["J06.9"],
        search_source="icd_fast_lookup",
    )
    events = iter_protocol_search_events(fb)
    assert len(events) == 1
    assert events[0]["search_source"] == "icd_fast_lookup"
    assert events[0]["has_icd"] is True
    assert events[0]["n_protocols"] == 1
    line = (fb / "protocol_search.jsonl").read_text(encoding="utf-8").strip()
    row = json.loads(line)
    assert row["event_type"] == "protocol_search"
