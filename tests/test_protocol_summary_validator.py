"""Tests for Protocol Summary validator."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from clinical_knowledge.protocol_summary.schema import ProtocolSummary, SummarySourceRef
from clinical_knowledge.protocol_summary.validator import (
    summary_is_usable,
    validate_protocol_summary,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "protocol_summaries" / "yaml"


@pytest.fixture(autouse=True)
def _root(monkeypatch, tmp_path):
    monkeypatch.setenv("PROTOCOL_SUMMARY_DATA_ROOT", str(FIX.parent))
    from clinical_knowledge.protocol_summary import config as cfg
    from clinical_knowledge.protocol_summary import loader

    cfg.protocol_summary_config = cfg.ProtocolSummaryConfig.from_env()
    loader.clear_protocol_summary_cache()


def _load(name: str) -> ProtocolSummary:
    data = yaml.safe_load((FIX / name).read_text(encoding="utf-8"))
    return ProtocolSummary.model_validate(data)


def test_valid_fixture_passes():
    s = _load("test_gastro_k30.yaml")
    r = validate_protocol_summary(s, strict=True)
    assert r.status in ("valid", "valid_with_warnings", "needs_human_review")
    assert summary_is_usable(s)


def test_missing_source_ref_fails():
    s = _load("test_gastro_k30.yaml")
    s.conditions[0].required_exams[0].source_ref = SummarySourceRef(protocol_id="test_gastro_k30")
    r = validate_protocol_summary(s, strict=True)
    assert r.status == "invalid"
    assert any(e.code in ("incomplete_source_ref", "missing_quote") for e in r.errors)


def test_missing_condition_name_fails():
    s = _load("test_gastro_k30.yaml")
    s.conditions[0].name = ""
    r = validate_protocol_summary(s)
    assert any(e.code == "missing_condition_name" for e in r.errors)
