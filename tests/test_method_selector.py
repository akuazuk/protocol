"""Tests for method selector."""
from __future__ import annotations

from pathlib import Path

import pytest

from clinical_knowledge.protocol_summary.method_selector import resolve_analysis_plan

FIX = Path(__file__).resolve().parent / "fixtures" / "protocol_summaries" / "yaml"


@pytest.fixture(autouse=True)
def _root(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_DATA_ROOT", str(FIX.parent))
    from clinical_knowledge.protocol_summary import config as cfg
    from clinical_knowledge.protocol_summary import loader

    cfg.protocol_summary_config = cfg.ProtocolSummaryConfig.from_env()
    loader.clear_protocol_summary_cache()


def test_legacy_when_disabled(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_ENABLED", "0")
    from clinical_knowledge.protocol_summary import config as cfg

    cfg.protocol_summary_config = cfg.ProtocolSummaryConfig.from_env()
    plan = resolve_analysis_plan(mode="hybrid")
    assert plan.use_legacy is True
    assert plan.use_summary is False


def test_summary_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_ENABLED", "1")
    monkeypatch.setenv("PROTOCOL_SUMMARY_MODE", "summary")
    monkeypatch.setenv("PROTOCOL_SUMMARY_FALLBACK_TO_LEGACY", "1")
    from clinical_knowledge.protocol_summary import config as cfg

    cfg.protocol_summary_config = cfg.ProtocolSummaryConfig.from_env()
    plan = resolve_analysis_plan(mode="summary", matched_protocol_ids=["nonexistent_protocol"])
    assert plan.use_legacy is True
    assert plan.fallback_to_legacy or plan.primary_source == "legacy"


def test_hybrid_with_valid_summary(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_ENABLED", "1")
    monkeypatch.setenv("PROTOCOL_SUMMARY_MODE", "hybrid")
    from clinical_knowledge.protocol_summary import config as cfg

    cfg.protocol_summary_config = cfg.ProtocolSummaryConfig.from_env()
    plan = resolve_analysis_plan(mode="hybrid", matched_protocol_ids=["test_gastro_k30"])
    assert plan.use_summary is True
    assert plan.use_legacy is True
