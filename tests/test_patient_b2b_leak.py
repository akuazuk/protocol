"""B2C invariant: B2B-поля (ЦИСЗ, send_gate, сырой structured) не утекают пациенту."""
from __future__ import annotations

from clinical_knowledge.patient_report import sanitize_patient_api_payload
from clinical_knowledge.patient_report_v2 import _B2B_FORBIDDEN_KEYS, strip_b2b_from_payload


def _payload_with_b2b() -> dict:
    return {
        "ok": True,
        "review_tier": "P1",
        "gate_score": 88,
        "send_gate": "ready",
        "cisz_readiness": {"ready": True},
        "structured_analysis": {"document": {"sections": {}}},
        "alignment": {"alignment_mean_score": 62},
        "criteria": [{"id": "c1"}],
        "review": {"limitations_ru": "internal"},
        "report_html": "<b>internal</b>",
        "report_markdown": "# internal",
        "_protocol_filter": {"removed_count": 1},
        "patient_report": {
            "traffic_light": "yellow",
            "overall_pct": 62,
            "gate_score": 88,
            "send_gate": "ready",
            "structured_analysis": {"raw": True},
            "alignment": {"x": 1},
            "review": {"secret": True},
            "_protocol_filter": {"removed_count": 1},
        },
    }


def test_sanitize_removes_b2b_top_and_nested():
    out = sanitize_patient_api_payload(_payload_with_b2b())
    pr = out.get("patient_report") or {}
    for key in _B2B_FORBIDDEN_KEYS:
        assert key not in out, f"B2B key {key} leaked at top level"
        assert key not in pr, f"B2B key {key} leaked into patient_report"


def test_sanitize_keeps_patient_fields():
    out = sanitize_patient_api_payload(_payload_with_b2b())
    pr = out.get("patient_report") or {}
    assert out["ok"] is True
    assert out["review_tier"] == "P1"
    assert pr["traffic_light"] == "yellow"
    assert pr["overall_pct"] == 62


def test_strip_b2b_is_pure_and_idempotent():
    payload = _payload_with_b2b()
    once = strip_b2b_from_payload(payload)
    twice = strip_b2b_from_payload(once)
    # исходный payload не мутируется (B2B-ключи на месте в оригинале)
    assert "gate_score" in payload
    for key in _B2B_FORBIDDEN_KEYS:
        assert key not in twice
        assert key not in (twice.get("patient_report") or {})
