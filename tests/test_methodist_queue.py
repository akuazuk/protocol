"""Тесты очереди методиста."""
from __future__ import annotations

from pathlib import Path

from clinical_knowledge.methodist_queue import build_methodist_queue

FEEDBACK_RENDER = Path(__file__).resolve().parents[1] / "data" / "ml" / "feedback_render"


def test_build_methodist_queue_from_render_export():
    if not FEEDBACK_RENDER.is_dir():
        return
    q = build_methodist_queue(limit=30)
    assert "priority" in q
    assert "pending" in q
    assert "counts" in q
    assert q["counts"]["total_kz"] >= 1
