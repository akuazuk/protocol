"""B2C catalog tier limits."""
from __future__ import annotations

from clinical_knowledge.patient_tier_gating import apply_catalog_tier_limits


def test_promo_tier_truncates_questions_and_citations():
    report = {
        "questions_for_doctor": ["q1", "q2", "q3", "q4", "q5"],
        "protocol_citations": [{"title": "КП", "quote": "текст"}],
        "blocks": [{"status": "ok"}, {"status": "concern"}],
    }
    out = apply_catalog_tier_limits(report, catalog_tier_id="promo")
    assert len(out["questions_for_doctor"]) == 3
    assert out.get("questions_truncated") is True
    assert out["protocol_citations"] == []
    assert out.get("tier_preview") is True
