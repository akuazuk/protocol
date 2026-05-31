"""Tests for Protocol Summary schema."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from clinical_knowledge.protocol_summary.schema import (
    ConditionSummary,
    ExamRequirement,
    ProtocolSummary,
    SummarySourceRef,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "protocol_summaries" / "yaml"


@pytest.fixture(autouse=True)
def _summary_data_root(monkeypatch):
    monkeypatch.setenv("PROTOCOL_SUMMARY_DATA_ROOT", str(FIX.parent))
    from clinical_knowledge.protocol_summary import config as cfg
    from clinical_knowledge.protocol_summary import loader

    cfg.protocol_summary_config = cfg.ProtocolSummaryConfig.from_env()
    loader.clear_protocol_summary_cache()


def test_protocol_summary_yaml_loads():
    import yaml

    data = yaml.safe_load((FIX / "test_gastro_k30.yaml").read_text(encoding="utf-8"))
    summary = ProtocolSummary.model_validate(data)
    assert summary.protocol_id == "test_gastro_k30"
    assert summary.conditions[0].icd10_codes == ["K30"]


def test_condition_requires_name_and_id():
    ref = SummarySourceRef(protocol_id="p1", page_start=1, quote="x")
    cond = ConditionSummary(
        condition_id="c1",
        name="Test",
        required_exams=[
            ExamRequirement(
                name="ОАК",
                requirement_level="required",
                source_ref=ref,
            ),
        ],
    )
    assert cond.name == "Test"


def test_extra_keys_ignored():
    data = {
        "protocol_id": "x",
        "unknown_field": 123,
        "source": {"title": "T", "local_path": "a.pdf"},
        "rubric": {"name": "R"},
        "conditions": [{"condition_id": "c", "name": "N", "icd10_codes": ["K30"]}],
    }
    s = ProtocolSummary.model_validate(data)
    assert s.protocol_id == "x"
