"""Nightly patient quality aggregation."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.patient_nightly_quality import aggregate_patient_feedback, build_markdown_report


def test_aggregate_empty(tmp_path: Path) -> None:
    fb = tmp_path / "feedback"
    fb.mkdir()
    stats = aggregate_patient_feedback(fb_root=fb)
    assert stats["review_count"] == 0


def test_build_markdown() -> None:
    stats = {"generated_at": "2026-06-24T00:00:00Z", "review_count": 2, "quality_flags": {"x": 1}}
    llm = {"summary_ru": "OK", "health_score": 4, "improvements": []}
    md = build_markdown_report(stats, llm, [])
    assert "Protocol B2C" in md
    assert "OK" in md


def test_build_methodist_view_empty(tmp_path: Path, monkeypatch) -> None:
    fb = tmp_path / "feedback"
    fb.mkdir()
    monkeypatch.setattr(
        "clinical_knowledge.patient_nightly_quality.aggregate_patient_feedback",
        lambda **kw: aggregate_patient_feedback(fb_root=fb),
    )
    from clinical_knowledge.patient_nightly_quality import build_methodist_patient_quality_view

    view = build_methodist_patient_quality_view()
    assert view["ok"] is True
    assert view["live_stats"]["review_count"] == 0
    assert "Protocol B2C" in view["markdown_ru"]
    assert view["llm_review"]["summary_ru"]
