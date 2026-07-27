"""Тесты аудита knowledge-корпуса (Workstream F ТЗ overnight-v1).

Аудит запускается на реальном корпусе summary; тест проверяет структуру и инварианты,
а не конкретные абсолютные числа (корпус может меняться).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from scripts.audit_kz_protocol_knowledge import run_audit


@pytest.fixture(scope="module")
def audit():
    return run_audit(mis_summary_path=None, queue_top=20)


def test_audit_has_core_metrics(audit):
    a = audit["audit"]
    assert a["protocols_total"] > 0
    for key in (
        "protocol_structured_coverage_pct",
        "penalty_eligible_coverage_pct",
        "source_verified_coverage_pct",
        "methodist_approved_coverage_pct",
        "protocols_without_safe_penalty_rule",
    ):
        assert key in a["metrics"]


def test_penalty_eligible_not_overstated(audit):
    # ключевой инвариант §10.2: наличие правила != пригодность к штрафу.
    a = audit["audit"]
    # без утверждённых методистом протоколов penalty-eligible не должно превышать
    # долю verified+approved. Approved=0 -> penalty-eligible=0.
    if a["review_status"].get("approved", 0) == 0:
        assert a["penalty_eligible_rules"] == 0
        assert a["metrics"]["penalty_eligible_coverage_pct"] == 0.0


def test_queue_sorted_by_priority(audit):
    q = audit["queue"]
    assert len(q) > 0
    priorities = [r["priority"] for r in q]
    assert priorities == sorted(priorities, reverse=True)
    for r in q:
        assert "penalty_ready_pct" in r
        assert "mis_frequency" in r


def test_coverage_block_present(audit):
    a = audit["audit"]
    for key in ("required_exams", "treatment", "red_flags", "follow_up"):
        assert key in a["coverage"]
