"""case_review_brief + gold fixture mo_1_test."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.mo_case_narrative import case_narrative_enabled, normalize_narrative
from clinical_knowledge.mo_case_review_brief import build_case_review_brief
from clinical_knowledge.mo_clinical_gaps import evaluate_mo_clinical_gaps, merge_clinical_gaps_into_findings

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mo_case_review_brief_mo1_gold.json"


def _gold() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_mo1_gold_gaps_and_brief() -> None:
    gold = _gold()
    clinical = gold["clinical"]
    findings = merge_clinical_gaps_into_findings([], clinical)
    codes = {f["code"] for f in findings}
    for code in gold["expect_gap_codes"]:
        assert code in codes, f"missing gap {code}"

    detail = {
        "zones": gold["zones_stub"],
        "findings": findings,
        "icd_visit_status": {"status": "ok", "label_ru": "МКБ ок"},
        "protocol_suggest": {"available": False},
        "patient_history": {"summary": {"n_visits": 0}},
        "record": {
            "visit_id": "mo_1_test",
            "clinical_diagnosis": clinical["clinical_diagnosis"],
            "diagnosis_code": "J45",
        },
    }
    brief = build_case_review_brief(detail)
    expect = gold["expect_brief"]
    assert brief["ok"] and brief["available"]
    assert brief["diagnosis_axes"]["methodology"]["band"] == expect["diagnosis_methodology_band"]
    assert brief["zones"]["plan"]["band"] == expect["plan_band"]
    assert len(brief["doctor_feedback"]) >= expect["min_doctor_feedback"]
    blob = " ".join(brief["doctor_feedback"]).lower()
    for needle in expect["must_mention_substrings"]:
        assert needle in blob, f"feedback missing {needle!r}: {brief['doctor_feedback']}"
    assert brief["decision_summary_ru"].startswith("•")
    assert "не путать" in (brief["icd"]["note_ru"] or "").lower() or "не" in (
        brief["diagnosis_axes"]["icd_directory"]["note_ru"] or ""
    ).lower()
    assert "слабо» из-за" in brief["zones"]["documentation"]["why_ru"] or "0.5" in brief[
        "zones"
    ]["documentation"]["why_ru"]


def test_zone_why_explains_high_pct_weak() -> None:
    brief = build_case_review_brief(
        {
            "zones": {
                "ok": True,
                "zone1": {"band": "weak", "pct": 91.7, "label_ru": "Оформление"},
                "zone2a": {"band": "ok", "pct": 100},
                "zone2b": {"band": "na", "kp_status": "unmatched"},
                "criteria": [
                    {
                        "zone": "documentation",
                        "title": "Факторы риска",
                        "score": 0.5,
                        "reason": "семейная АГ",
                    }
                ],
            },
            "findings": [],
            "record": {},
        }
    )
    why = brief["zones"]["documentation"]["why_ru"]
    assert "91.7" in why
    assert "слабо" in why
    assert "из-за" in why


def test_narrative_default_off() -> None:
    assert case_narrative_enabled() is False
    norm = normalize_narrative(
        {
            "summary_ru": "тест",
            "clinical_gaps_ru": ["разрыв"],
            "doctor_feedback_ru": ["врачу"],
            "confidence": 0.7,
        }
    )
    assert norm["available"] is True
    assert norm["doctor_feedback_ru"] == ["врачу"]


def test_frontend_has_review_brief_section() -> None:
    mo_app = Path(__file__).resolve().parents[1] / "frontend" / "web" / "shared" / "mo-app.js"
    text = mo_app.read_text(encoding="utf-8")
    assert "Итог разбора" in text
    assert "renderReviewBrief" in text
    assert "review-brief-prefill" in text
    assert "Подставить в решение методиста" in text
