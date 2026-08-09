from __future__ import annotations

from clinical_knowledge.mo_shadow_dx_plan import (
    attention_band_for_endpoint,
    build_shadow_payload,
    case_has_shadow_attention,
    summarize_shadow_for_ui,
)


def test_partial_and_good_are_not_attention() -> None:
    for verdict in ("good", "partial", "blocked", "na", ""):
        band = attention_band_for_endpoint(
            endpoint="dx",
            verdict=verdict,
            score_pct=20,
            potential_harm=False,
        )
        assert band["band"] == "none"


def test_poor_requires_score_at_most_45() -> None:
    soft = attention_band_for_endpoint(
        endpoint="dx",
        verdict="poor",
        score_pct=80,
        potential_harm=False,
    )
    assert soft["band"] == "none"
    assert soft["softened"] is True
    ok = attention_band_for_endpoint(
        endpoint="dx",
        verdict="poor",
        score_pct=40,
        potential_harm=False,
    )
    assert ok["band"] == "poor"


def test_critical_with_harm_allows_up_to_45() -> None:
    band = attention_band_for_endpoint(
        endpoint="dx",
        verdict="critical",
        score_pct=40,
        potential_harm=True,
    )
    assert band["band"] == "critical"
    high = attention_band_for_endpoint(
        endpoint="dx",
        verdict="critical",
        score_pct=70,
        potential_harm=True,
    )
    assert high["band"] == "none"


def test_poor_plus_harm_becomes_critical() -> None:
    band = attention_band_for_endpoint(
        endpoint="plan",
        verdict="poor",
        score_pct=25,
        potential_harm=True,
    )
    assert band["band"] == "critical"


def test_plan_ensemble_can_only_downgrade_poor() -> None:
    band = attention_band_for_endpoint(
        endpoint="plan",
        verdict="poor",
        score_pct=40,
        potential_harm=False,
        plan_ensemble_pct=70,
    )
    assert band["band"] == "none"
    assert band["soften_reason"] == "plan_ensemble_downgrade"
    critical = attention_band_for_endpoint(
        endpoint="plan",
        verdict="critical",
        score_pct=20,
        potential_harm=False,
        plan_ensemble_pct=90,
    )
    assert critical["band"] == "critical"


def test_build_payload_and_ui_summary() -> None:
    payload = build_shadow_payload(
        case_id="3643940",
        visit_date="2026-08-08",
        model="gemini-3.6-flash",
        dx_result={
            "verdict": "poor",
            "dx_evidence_pct": 35,
            "potential_harm": False,
            "summary_ru": "Диагноз слабо подтверждён обследованиями",
            "icd_fit": "partial",
        },
        plan_result={
            "verdict": "partial",
            "plan_protocol_pct": 70,
            "potential_harm": False,
            "summary_ru": "План частично соответствует КП",
            "provenance": "kp_grounded",
        },
        clinical_concordance_pct=80,
    )
    assert payload["case_attention_band"] == "poor"
    assert payload["dx"]["attention"]["band"] == "poor"
    assert payload["plan"]["attention"]["band"] == "none"
    assert payload["plan"]["ensemble_pct"] == 75.0
    ui = summarize_shadow_for_ui(payload)
    assert ui["available"] is True
    assert ui["shadow"] is True
    assert ui["official_score"] is False
    assert "не официальная" in ui["disclaimer_ru"]
    assert case_has_shadow_attention(ui) is True
