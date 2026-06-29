"""Doctor-facing protocol search: summary-first ordering + search feedback telemetry."""
from __future__ import annotations

import pytest

from clinical_knowledge.search_retrieval import _order_path_boost
from clinical_knowledge.search_telemetry import aggregate_search_feedback, log_search_feedback


def test_order_path_boost_summary_first():
    target = ["minzdrav_protocols/a.pdf", "minzdrav_protocols/b.pdf"]
    summary = ["minzdrav_protocols/sum.pdf", "minzdrav_protocols/a.pdf"]
    out = _order_path_boost(target, summary, summary_first=True)
    assert out[0] == "minzdrav_protocols/sum.pdf"
    # dedupe preserved
    assert out.count("minzdrav_protocols/a.pdf") == 1


def test_order_path_boost_target_first_default():
    target = ["minzdrav_protocols/a.pdf"]
    summary = ["minzdrav_protocols/sum.pdf"]
    out = _order_path_boost(target, summary, summary_first=False)
    assert out[0] == "minzdrav_protocols/a.pdf"
    assert "minzdrav_protocols/sum.pdf" in out


def test_log_search_feedback_rejects_bad_verdict():
    with pytest.raises(ValueError):
        log_search_feedback(query="орви", verdict="maybe")


def test_aggregate_search_feedback_counts_and_top_miss():
    events = [
        {"verdict": "fit", "ts": "2026-06-29T10:00:00Z", "rejected_basename": ""},
        {"verdict": "miss", "ts": "2026-06-29T10:01:00Z", "rejected_basename": "wrong.pdf"},
        {"verdict": "miss", "ts": "2026-06-29T10:02:00Z", "rejected_basename": "wrong.pdf"},
    ]
    agg = aggregate_search_feedback(events)
    assert agg["total"] == 3
    assert agg["fit"] == 1
    assert agg["miss"] == 2
    assert agg["fit_pct"] == pytest.approx(33.3, abs=0.2)
    assert agg["top_miss_protocols"][0] == {"basename": "wrong.pdf", "count": 2}
    assert agg["recent"][0]["ts"] == "2026-06-29T10:02:00Z"
    assert agg["readiness"]["target_golden"] == 20
