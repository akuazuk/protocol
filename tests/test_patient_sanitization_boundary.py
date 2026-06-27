"""B2B boundary sanitization for patient API."""
from __future__ import annotations

from clinical_knowledge.patient_report import sanitize_patient_api_payload


def test_sanitize_strips_alignment_and_gate() -> None:
    payload = {
        "ok": True,
        "send_gate": {"allowed": False},
        "gate_score": 42,
        "structured_analysis": {"matches": []},
        "alignment": {"alignment_cards": []},
        "review": {"criteria": []},
        "patient_report": {
            "traffic_light": "yellow",
            "gate_score": 1,
            "alignment": {},
        },
    }
    out = sanitize_patient_api_payload(payload)
    assert "send_gate" not in out
    assert "structured_analysis" not in out
    assert "alignment" not in out
    assert "review" not in out
    pr = out["patient_report"]
    assert "gate_score" not in pr
    assert "alignment" not in pr
