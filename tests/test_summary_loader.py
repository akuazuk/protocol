"""Tests for protocol summary loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from clinical_knowledge.protocol_summary.loader import (
    find_conditions_by_icd,
    find_conditions_by_text,
    load_summary_by_protocol_id,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "protocol_summaries" / "yaml"


@pytest.fixture(autouse=True)
def _root(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_DATA_ROOT", str(FIX.parent))
    from clinical_knowledge.protocol_summary import config as cfg
    from clinical_knowledge.protocol_summary import loader

    cfg.protocol_summary_config = cfg.ProtocolSummaryConfig.from_env()
    loader.clear_protocol_summary_cache()


def test_load_by_protocol_id():
    s = load_summary_by_protocol_id("test_gastro_k30")
    assert s is not None
    assert s.conditions[0].condition_id == "k30_functional_dyspepsia"


def test_find_by_icd():
    found = find_conditions_by_icd("K30")
    assert any(c.condition_id == "k30_functional_dyspepsia" for c in found)


def test_find_by_text():
    found = find_conditions_by_text("диспепсия")
    assert len(found) >= 1
