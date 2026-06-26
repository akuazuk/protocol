"""Tests for B2C waves A–D modules."""
from __future__ import annotations

from clinical_knowledge.patient_analytics import record_patient_event
from clinical_knowledge.patient_clinic_config import resolve_clinic, resolve_tier
from clinical_knowledge.patient_payment import create_payment_session, verify_payment_token
from clinical_knowledge.patient_p2_enrich import enrich_patient_report_p2
from clinical_knowledge.patient_report import build_patient_report


def test_patient_report_schema_v2_fields() -> None:
    l1 = {
        "confidence_score": 85,
        "matched_protocols_count": 1,
        "alignment": {
            "alignment_mean_score": 72,
            "alignment_cards": [
                {
                    "block_id": "diagnosis",
                    "name_ru": "Диагноз",
                    "score_pct": 80,
                    "comment_ru": "Код МКБ указан.",
                    "gaps_ru": [],
                    "protocol_excerpt": "Указывают код МКБ.",
                },
            ],
        },
    }
    rep = build_patient_report(l1)
    assert rep["report_schema_version"] == 2
    assert rep["headline_ru"]
    assert rep["blocks"][0]["why_ru"]


def test_low_confidence_caps_traffic_light() -> None:
    l1 = {
        "confidence_score": 40,
        "alignment": {
            "alignment_mean_score": 85,
            "alignment_cards": [],
            "limitations_ru": "Текст не читается.",
        },
    }
    rep = build_patient_report(l1)
    assert rep["traffic_light"] in ("yellow", "red")


def test_p2_enrich_adds_narratives() -> None:
    base = build_patient_report(
        {
            "confidence_score": 70,
            "alignment": {
                "alignment_mean_score": 55,
                "alignment_cards": [
                    {
                        "block_id": "treatment",
                        "name_ru": "Лечение",
                        "score_pct": 40,
                        "comment_ru": "Нет дозировки.",
                        "gaps_ru": ["доза"],
                        "protocol_excerpt": "Указывают дозу препарата.",
                    },
                ],
            },
        }
    )
    p2 = enrich_patient_report_p2(base)
    assert p2["plain_narratives"]
    assert p2["review_tier_product"] == "P2"


def test_clinic_and_tier_resolve() -> None:
    c = resolve_clinic("kravira")
    assert c and c["name_ru"]
    t = resolve_tier("plus", c)
    assert t["price_byn"] == 6.99


def test_payment_dev_session() -> None:
    sess = create_payment_session(tier_id="basic")
    assert sess["payment_token"].startswith("dev-")
    assert verify_payment_token(sess["payment_token"])


def test_analytics_allowed_event() -> None:
    out = record_patient_event(event="report_view", meta={"light": "green", "pct": 72})
    assert out["ok"] is True


def test_analytics_rejects_pii_event() -> None:
    out = record_patient_event(event="unknown_event")
    assert out["ok"] is False
