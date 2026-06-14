"""GET /api/methodist/analysis/{id} и protocol_match stats."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.feedback_store import save_analysis_snapshot
from clinical_knowledge.methodist_analysis import get_methodist_analysis
from clinical_knowledge.methodist_stats import _compute_protocol_match_stats


def test_get_methodist_analysis_roundtrip(tmp_path: Path, monkeypatch):
    from clinical_knowledge import feedback_store

    monkeypatch.setattr(feedback_store, "analyses_dir", lambda: tmp_path / "analyses")
    monkeypatch.setattr(feedback_store, "secure_kz_dir", lambda: tmp_path / "secure")

    aid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    snap = {
        "analysis_id": aid,
        "text_hash": "sha256:abc",
        "tier": "L1",
        "saved_at": "2026-06-01T12:00:00Z",
        "text_excerpt": "Диагноз I80.1",
        "api_result": {"review": {"overall_compliance_pct": 55}, "analysis_id": aid},
    }
    save_analysis_snapshot(aid, snap)
    (tmp_path / "secure").mkdir(parents=True)
    (tmp_path / "secure" / "abc.txt").write_text("Полный текст КЗ", encoding="utf-8")

    out = get_methodist_analysis(aid)
    assert out is not None
    assert out["analysis_id"] == aid
    assert out["api_result"]["review"]["overall_compliance_pct"] == 55
    assert out["has_full_text"] is True
    assert "Полный текст" in out.get("full_text", "")


def test_get_methodist_analysis_missing():
    assert get_methodist_analysis("00000000-0000-0000-0000-000000000000") is None


def test_protocol_match_stats_hit_at_k():
    kz = [
        {
            "event_type": "kz_analysis",
            "analysis_id": "a1",
            "retrieval_top_paths": ["gastro/a.pdf", "cardio/b.pdf", "derma/c.pdf"],
            "matched_protocol_paths": ["gastro/a.pdf"],
        }
    ]
    reviews = [
        {
            "event_type": "analysis_review",
            "analysis_id": "a1",
            "tags": ["wrong_protocol"],
            "retrieval_fix": {
                "chosen_path": "derma/c.pdf",
                "rejected_path": "gastro/a.pdf",
            },
        }
    ]
    stats = _compute_protocol_match_stats(kz, reviews, [])
    assert stats["labeled_retrieval_gold"] == 1
    assert stats["protocol_hit_at_1_pct"] == 0.0
    assert stats["protocol_hit_at_3_pct"] == 100.0
    assert stats["protocol_tag_reviews"] == 1
