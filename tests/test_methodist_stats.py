"""Тесты агрегации статистики дашборда методиста."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.methodist_stats import build_methodist_dashboard_stats

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_RENDER = ROOT / "data" / "ml" / "feedback_render"


def test_build_methodist_dashboard_stats_from_render_export():
    if not FEEDBACK_RENDER.is_dir():
        return
    stats = build_methodist_dashboard_stats(feedback_dir=FEEDBACK_RENDER)
    assert stats["summary"]["total_events"] >= 1
    assert "unique_kz" in stats["summary"]
    assert stats["pool"]["in_training_pool"] >= 1
    assert isinstance(stats["ml_readiness"]["items"], list)
    assert len(stats["ml_readiness"]["items"]) >= 5
    assert "charts" in stats
    assert "specialties" in stats
    assert isinstance(stats["engine_releases"], list)


def test_readiness_pct_bounded():
    if not FEEDBACK_RENDER.is_dir():
        return
    stats = build_methodist_dashboard_stats(feedback_dir=FEEDBACK_RENDER)
    for item in stats["ml_readiness"]["items"]:
        assert 0 <= item["pct"] <= 100
        assert item["current"] <= item["target"] or item["pct"] == 100.0


def test_reanalysis_deltas_structure():
    if not FEEDBACK_RENDER.is_dir():
        return
    stats = build_methodist_dashboard_stats(feedback_dir=FEEDBACK_RENDER)
    for row in stats.get("reanalysis_deltas") or []:
        assert "text_hash_short" in row
        assert "before_overall_pct" in row
        assert "after_overall_pct" in row


def test_empty_feedback_dir():
    stats = build_methodist_dashboard_stats(feedback_dir=Path("/tmp/nonexistent_feedback_xyz"))
    assert stats["summary"]["total_events"] == 0
    assert stats["summary"]["unique_kz"] == 0
