"""Integration: summary/hybrid modes activate summary rules."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from clinical_knowledge.consult_analysis import analyze_consultation_text

FIX = Path(__file__).resolve().parent / "fixtures" / "consultations"
DATA_ROOT = Path(__file__).resolve().parent / "fixtures" / "protocol_summaries"


@pytest.fixture(autouse=True)
def _summary_env(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_DATA_ROOT", str(DATA_ROOT))
    from clinical_knowledge.protocol_summary import config as cfg
    from clinical_knowledge.protocol_summary import loader

    cfg.protocol_summary_config = cfg.ProtocolSummaryConfig.from_env()
    loader.clear_protocol_summary_cache()


def test_summary_mode_uses_summary_rules_for_k30(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_ENABLED", "1")
    text = (FIX / "gastro_1.txt").read_text(encoding="utf-8")
    res = analyze_consultation_text(text, with_markdown=False, analysis_mode="summary")
    comp = res["compliance"]
    assert comp.get("analysis_mode") == "summary"
    assert comp.get("protocol_summary_used") is True
    assert comp.get("summary_result_available") is True
    ev = comp.get("evidence_map") or []
    assert any(e.get("rule_source") == "summary" for e in ev)


def test_hybrid_shows_diagnostics(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_ENABLED", "1")
    text = (FIX / "gastro_1.txt").read_text(encoding="utf-8")
    res = analyze_consultation_text(text, with_markdown=False, analysis_mode="hybrid")
    comp = res["compliance"]
    assert comp.get("analysis_mode") == "hybrid"
    assert comp.get("protocol_summary_used") is True
    assert comp.get("summary_diagnostics")


def test_legacy_json_has_mode_legacy(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_ENABLED", "0")
    text = (FIX / "gastro_1.txt").read_text(encoding="utf-8")
    comp = analyze_consultation_text(text, with_markdown=False, analysis_mode="legacy")["compliance"]
    assert comp.get("analysis_mode") == "legacy"
    assert comp.get("protocol_summary_used") is False


def test_summary_not_found_is_explicit(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_ENABLED", "1")
    monkeypatch.setenv("PROTOCOL_SUMMARY_FALLBACK_TO_LEGACY", "0")
    text = (FIX / "mg_1.txt").read_text(encoding="utf-8")
    comp = analyze_consultation_text(text, with_markdown=False, analysis_mode="summary")["compliance"]
    assert comp.get("analysis_mode") == "summary"
    # J06 — нет test fixture summary
    if not comp.get("protocol_summary_used"):
        assert comp.get("summary_diagnostics") or comp.get("fallback_to_legacy") is not None
