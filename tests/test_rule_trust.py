"""Тесты уровней доверия к правилам (Workstream B ТЗ overnight-v1)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clinical_knowledge.rule_trust import (
    TRUST_A,
    TRUST_B,
    TRUST_C,
    TRUST_D,
    can_hard_gate,
    penalty_allowed,
    rule_trust_diagnostics,
    trust_for_rule,
)

_QUOTE = "общий анализ крови развёрнутый обязателен при данном состоянии"


def test_approved_methodist_is_A_and_penalizes():
    rule = {"rule_source": "summary", "review_status": "approved", "source": {"quote": _QUOTE}}
    info = trust_for_rule(rule)
    assert info.trust_level == TRUST_A
    assert info.penalty_allowed is True
    assert penalty_allowed(info.trust_level)
    assert can_hard_gate(info.trust_level)


def test_reviewed_with_quote_is_B():
    rule = {"rule_source": "summary", "review_status": "reviewed", "source": {"quote": _QUOTE}}
    info = trust_for_rule(rule)
    assert info.trust_level == TRUST_B
    assert info.penalty_allowed is True


def test_auto_summary_without_review_is_C_advisory():
    rule = {"rule_source": "summary", "review_status": "not_reviewed", "source": {"quote": _QUOTE}}
    info = trust_for_rule(rule)
    assert info.trust_level == TRUST_C
    assert info.penalty_allowed is False
    assert not penalty_allowed(info.trust_level)
    assert not can_hard_gate(info.trust_level)


def test_rich_table_and_path_are_D_heuristic():
    for src in ("rich_table", "path_template", "fallback"):
        info = trust_for_rule({"rule_source": src, "source": {"quote": _QUOTE}})
        assert info.trust_level == TRUST_D
        assert info.penalty_allowed is False


def test_ab_without_quote_downgraded_to_C():
    # заявлен reviewed, но без цитаты -> не может штрафовать (§6.3)
    info = trust_for_rule({"rule_source": "summary", "review_status": "reviewed", "source": {"quote": ""}})
    assert info.trust_level == TRUST_C
    assert info.penalty_allowed is False


def test_no_auto_upgrade_c_to_b():
    # auto источник без review не должен подняться до B даже с цитатой
    info = trust_for_rule({"rule_source": "auto_extracted", "source": {"quote": _QUOTE}})
    assert info.trust_level == TRUST_C


def test_diagnostics_counts():
    rules = [
        {"rule_source": "summary", "review_status": "approved", "source": {"quote": _QUOTE}},  # A
        {"rule_source": "summary", "review_status": "not_reviewed", "source": {"quote": _QUOTE}},  # C
        {"rule_source": "rich_table", "source": {"quote": _QUOTE}},  # D
    ]
    d = rule_trust_diagnostics(rules)
    assert d["rules_total"] == 3
    assert d["rules_penalty_eligible"] == 1
    assert d["rules_advisory"] == 1
    assert d["rules_heuristic"] == 1
