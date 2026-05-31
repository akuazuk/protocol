"""Tests for summary → rules conversion."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from clinical_knowledge.protocol_summary.schema import ProtocolSummary
from clinical_knowledge.protocol_summary.summary_to_rules import (
    protocol_rule_to_legacy_dict,
    summary_to_protocol_rules,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "protocol_summaries" / "yaml"


def _load(name: str) -> ProtocolSummary:
    data = yaml.safe_load((FIX / name).read_text(encoding="utf-8"))
    return ProtocolSummary.model_validate(data)


def test_required_exam_becomes_rule():
    rules = summary_to_protocol_rules(_load("test_gastro_k30.yaml"))
    exam_rules = [r for r in rules if r.rule_type == "required_exam_rule"]
    assert exam_rules
    assert exam_rules[0].rule_source == "summary"
    legacy = protocol_rule_to_legacy_dict(exam_rules[0])
    assert legacy["rule_type"] == "required_exam"
    assert legacy["exam"] == "ЭГДС"


def test_red_flag_rule():
    rules = summary_to_protocol_rules(_load("test_gastro_k30.yaml"))
    rf = [r for r in rules if r.rule_type == "red_flag_rule"]
    assert rf
    assert rf[0].generated_from_summary is True


def test_drug_rule_from_phleb():
    rules = summary_to_protocol_rules(_load("test_phleb_i801.yaml"))
    drug = [r for r in rules if "ривароксабан" in " ".join(r.expected_items).lower()]
    assert drug
