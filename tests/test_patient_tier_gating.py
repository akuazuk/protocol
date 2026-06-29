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


def test_promo_truncates_action_checklist_with_count():
    report = {
        "questions_for_doctor": ["q1", "q2", "q3", "q4", "q5"],
        "action_checklist": [{"text": "q" + str(i)} for i in range(5)],
        "protocol_summary_panel": {"items": [{"name_ru": "ОАК", "present": True}]},
        "blocks": [{"status": "concern"}],
    }
    out = apply_catalog_tier_limits(report, catalog_tier_id="promo")
    assert len(out["action_checklist"]) == 3
    assert out["questions_hidden_count"] == 2
    assert out["questions_total"] == 5
    # позитивная панель протокола доступна только с цитатами (не на промо)
    assert "protocol_summary_panel" not in out


def test_plus_tier_inherits_questions_and_citations():
    report = {
        "questions_for_doctor": ["q1", "q2", "q3", "q4", "q5"],
        "action_checklist": [{"text": "q" + str(i)} for i in range(5)],
        "protocol_citations": [{"title": "КП", "quote": "текст"}],
        "protocol_summary_panel": {"items": [{"name_ru": "ОАК", "present": True}]},
        "blocks": [{"status": "ok"}, {"status": "concern"}],
    }
    out = apply_catalog_tier_limits(report, catalog_tier_id="plus")
    # plus наследует questions+citations через раскрытие includes
    assert len(out["questions_for_doctor"]) == 5
    assert out.get("questions_truncated") is not True
    assert out["protocol_citations"]
    assert "protocol_summary_panel" in out
    assert out.get("tier_preview") is False
