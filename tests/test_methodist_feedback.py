"""Tests for Methodist Workbench feedback store and API (phase A)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_knowledge.feedback_store import (
    append_feedback_event,
    build_kz_analysis_event,
    enrich_result_with_methodist_autolog,
    expand_analysis_review_events,
    text_hash,
    validate_and_normalize_event,
)


@pytest.fixture
def feedback_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fb = tmp_path / "feedback"
    secure = tmp_path / "secure" / "kz_text"
    analyses = tmp_path / "analyses"
    monkeypatch.setenv("ML_FEEDBACK_DIR", str(fb))
    monkeypatch.setenv("METHODIST_TOKEN", "test-methodist-token")
    monkeypatch.setattr(
        "clinical_knowledge.feedback_store.secure_kz_dir",
        lambda: secure,
    )
    monkeypatch.setattr(
        "clinical_knowledge.feedback_store.analyses_dir",
        lambda: analyses,
    )
    return {"feedback": fb, "secure": secure, "analyses": analyses}


def test_text_hash_stable():
    h1 = text_hash("Жалобы: изжога\n")
    h2 = text_hash("Жалобы: изжога")
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_append_kz_analysis(feedback_env):
    event = {
        "event_type": "kz_analysis",
        "analysis_id": "a1",
        "text_hash": "sha256:abc",
        "tier": "L0",
    }
    eid = append_feedback_event(event)
    assert eid
    path = feedback_env["feedback"] / "kz_analysis.jsonl"
    assert path.is_file()
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["analysis_id"] == "a1"


def test_expand_analysis_review_creates_override_events():
    review = {
        "event_type": "analysis_review",
        "analysis_id": "a1",
        "text_hash": "sha256:x",
        "rating": 2,
        "verdict": "partially_wrong",
        "reviewer": "М.М.",
        "overrides": [
            {
                "rule_id": "required_exam_egds",
                "system_pass": True,
                "human_pass": False,
                "note": "ложное срабатывание",
            }
        ],
        "retrieval_fix": {
            "query": "ГЭРБ",
            "rejected_path": "gastro/a.pdf",
            "chosen_path": "gastro/b.pdf",
        },
    }
    events = expand_analysis_review_events(review)
    types = [e["event_type"] for e in events]
    assert types.count("analysis_review") == 1
    assert "methodist_override" in types
    assert "retrieval_fix" in types


def test_enrich_result_with_autolog(feedback_env):
    result = {
        "ok": True,
        "send_gate": {"gate_score": 70, "sign_decision": "allowed_with_warnings"},
        "structured_analysis": {"compliance": {"overall_status": "needs_review"}},
    }
    out = enrich_result_with_methodist_autolog(
        result,
        tier="L0",
        full_text="Диагноз: K21.9",
        consultation_id="t1",
        latency_ms=100,
    )
    assert out.get("analysis_id")
    assert out.get("text_hash", "").startswith("sha256:")
    assert (feedback_env["secure"] / (out["text_hash"].split(":")[-1] + ".txt")).is_file()
    assert (feedback_env["analyses"] / f"{out['analysis_id']}.json").is_file()


def test_validate_analysis_review_requires_fields():
    with pytest.raises(ValueError, match="rating"):
        validate_and_normalize_event(
            {"event_type": "analysis_review", "verdict": "correct", "reviewer": "x"}
        )


def test_build_kz_analysis_from_pipeline_result():
    result = {
        "review_tier": "L1",
        "send_gate": {"gate_score": 55, "sign_decision": "review_required"},
        "clinical_rules": {
            "rules_check": {
                "rules_compliance_pct": 40.0,
                "findings": [
                    {"rule_id": "population_mismatch", "passed": False},
                ],
            }
        },
        "retrieval_paths": ["gastroenterologiya/kp.pdf"],
    }
    ev = build_kz_analysis_event(
        result=result,
        tier="L1",
        full_text="Тест КЗ",
        latency_ms=50,
    )
    assert ev["event_type"] == "kz_analysis"
    assert ev["failed_rule_ids"] == ["population_mismatch"]
    assert ev["rules_compliance_pct"] == 40.0


def test_api_ml_feedback_forbidden_without_token(feedback_env):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import rag_server

    client = TestClient(rag_server.app)
    r = client.post("/api/ml/feedback", json={"event_type": "retrieval_fix", "reviewer": "x"})
    assert r.status_code == 403


def test_api_ml_feedback_ok_with_token(feedback_env):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import rag_server

    client = TestClient(rag_server.app)
    headers = {"X-Methodist-Token": "test-methodist-token", "X-Methodist-Reviewer": "I.I."}
    body = {
        "event_type": "analysis_review",
        "analysis_id": "uuid-1",
        "text_hash": "sha256:deadbeef",
        "rating": 4,
        "verdict": "mostly_correct",
        "reviewer": "I.I.",
    }
    r = client.post("/api/ml/feedback", json=body, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert (feedback_env["feedback"] / "analysis_review.jsonl").is_file()


def test_consult_screen_autolog_with_token(feedback_env, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import rag_server

    sample = (Path(__file__).parent / "fixtures" / "consultations" / "gastro_adult.txt").read_text(
        encoding="utf-8"
    )
    client = TestClient(rag_server.app)
    headers = {"X-Methodist-Token": "test-methodist-token"}
    r = client.post(
        "/api/consult-compliance-screen",
        json={"text": sample, "consultation_id": "test", "methodist_mode": True},
        headers=headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("analysis_id")
    assert (feedback_env["feedback"] / "kz_analysis.jsonl").is_file()
