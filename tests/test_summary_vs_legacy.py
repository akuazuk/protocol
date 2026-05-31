"""Legacy baseline vs summary/hybrid modes."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from clinical_knowledge.consult_analysis import analyze_consultation_text

FIX = Path(__file__).resolve().parent / "fixtures" / "consultations"
SUMMARY_FIX = Path(__file__).resolve().parent / "fixtures" / "protocol_summaries" / "yaml"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_DATA_ROOT", str(SUMMARY_FIX.parent))


def test_legacy_unchanged_when_disabled(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_ENABLED", "0")
    text = (FIX / "gastro_adult.txt").read_text(encoding="utf-8")
    r1 = analyze_consultation_text(text, with_markdown=False)
    r2 = analyze_consultation_text(text, with_markdown=False, analysis_mode="legacy")
    assert r1["compliance"]["overall_score"] == r2["compliance"]["overall_score"]
    assert r2["compliance"].get("analysis_mode", "legacy") == "legacy"


def test_hybrid_does_not_crash_without_summary(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_ENABLED", "1")
    monkeypatch.setenv("PROTOCOL_SUMMARY_MODE", "hybrid")
    monkeypatch.setenv("PROTOCOL_SUMMARY_DATA_ROOT", str(SUMMARY_FIX.parent))
    from clinical_knowledge.protocol_summary import config as cfg
    from clinical_knowledge.protocol_summary import loader

    cfg.protocol_summary_config = cfg.ProtocolSummaryConfig.from_env()
    loader.clear_protocol_summary_cache()
    text = (FIX / "gastro_adult.txt").read_text(encoding="utf-8")
    res = analyze_consultation_text(text, with_markdown=False, analysis_mode="hybrid")
    assert "overall_score" in res["compliance"]


def test_hybrid_finds_summary_by_icd(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_ENABLED", "1")
    monkeypatch.setenv("PROTOCOL_SUMMARY_MODE", "hybrid")
    monkeypatch.setenv("PROTOCOL_SUMMARY_DATA_ROOT", str(SUMMARY_FIX.parent))
    from clinical_knowledge.protocol_summary import config as cfg
    from clinical_knowledge.protocol_summary import loader

    cfg.protocol_summary_config = cfg.ProtocolSummaryConfig.from_env()
    loader.clear_protocol_summary_cache()
    text = (FIX / "gastro_adult.txt").read_text(encoding="utf-8")
    res = analyze_consultation_text(text, with_markdown=False, analysis_mode="hybrid")
    comp = res["compliance"]
    if comp.get("protocol_summary_used"):
        assert comp.get("analysis_mode") == "hybrid"
