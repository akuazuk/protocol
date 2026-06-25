"""Тесты экспорта PlanDefinition из summary cards."""
from __future__ import annotations

from pathlib import Path

import pytest

from clinical_knowledge.protocol_summary.loader import load_summary_by_protocol_id
from clinical_knowledge.protocol_summary.plan_definition_export import (
    export_summaries_to_plan_definitions,
    summary_to_plan_definition,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "protocol_summaries" / "yaml"


@pytest.fixture(autouse=True)
def _root(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_DATA_ROOT", str(FIX.parent))
    from clinical_knowledge.protocol_summary import config as cfg
    from clinical_knowledge.protocol_summary import loader

    cfg.protocol_summary_config = cfg.ProtocolSummaryConfig.from_env()
    loader.clear_protocol_summary_cache()


def test_summary_to_plan_definition_minimal():
    summary = load_summary_by_protocol_id("test_gastro_k30")
    assert summary is not None
    pd = summary_to_plan_definition(summary)
    assert pd["resourceType"] == "PlanDefinition"
    assert pd["title"]
    assert isinstance(pd.get("action"), list)
    assert len(pd["action"]) >= 1
    assert pd["extension"]
    codes = pd.get("subjectCodeableConcept", {}).get("coding") or []
    assert any(c.get("code", "").startswith("K") for c in codes)


def test_export_batch_skips_empty():
    summary = load_summary_by_protocol_id("test_gastro_k30")
    assert summary is not None
    batch = export_summaries_to_plan_definitions([summary], usable_only=False)
    assert len(batch) == len(summary.conditions)
