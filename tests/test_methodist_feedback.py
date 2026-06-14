"""Tests for Methodist Workbench feedback store and API (phase A)."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from clinical_knowledge.feedback_store import (
    append_feedback_event,
    build_feedback_export_tar_gz,
    build_kz_analysis_event,
    enrich_result_with_methodist_autolog,
    expand_analysis_review_events,
    feedback_dir,
    text_hash,
    validate_and_normalize_event,
)
from clinical_knowledge.rule_labels_ru import rule_title_ru


GERD_FORMULA_RULE_ID = "9f9e0fb1_auto_gerd_diagnosis_formula"
GERD_EXAM_RULE_ID = "required_exam_egds"
POPULATION_RULE_ID = "gerd_population_guard"


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
                "rule_id": GERD_EXAM_RULE_ID,
                "system_pass": True,
                "human_pass": False,
                "note": "ЭГДС указана в КЗ сокращённо - система ошибочно требует повторно",
            }
        ],
        "retrieval_fix": {
            "query": "ГЭРБ изжога",
            "rejected_path": "gastro/a.pdf",
            "chosen_path": "gastro/b.pdf",
        },
    }
    events = expand_analysis_review_events(review)
    types = [e["event_type"] for e in events]
    assert types.count("analysis_review") == 1
    assert "methodist_override" in types
    assert "retrieval_fix" in types
    override = next(e for e in events if e["event_type"] == "methodist_override")
    assert override["rule_id"] == GERD_EXAM_RULE_ID
    assert "ЭГДС" in override["note"]


def test_rule_title_ru_for_gerd_diagnosis_formula():
    title = rule_title_ru(
        GERD_FORMULA_RULE_ID,
        {
            "rule_type": "diagnosis_formula",
            "message_ru": "В формулировке диагноза не хватает компонентов: этиология",
        },
    )
    assert "ГЭРБ" in title
    assert "диагноз" in title.lower()
    assert "9f9e0fb1" not in title
    assert "gerd_diagnosis_formula" not in title


def test_rule_title_ru_for_required_exam():
    title = rule_title_ru(GERD_EXAM_RULE_ID, {"rule_type": "required_exam", "exam": "ФГДС"})
    assert "ФГДС" in title
    assert "обследование" in title.lower()


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
    assert out.get("methodist_tier_meta", {}).get("label_ru", "").startswith("L0")
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
                    {
                        "rule_id": POPULATION_RULE_ID,
                        "rule_type": "population_mismatch",
                        "passed": False,
                        "title_ru": "Несоответствие возрастной группе: ГЭРБ",
                        "message_ru": "Протокол для детей, в КЗ указаны взрослые.",
                    },
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
    assert ev["failed_rule_ids"] == [POPULATION_RULE_ID]
    assert ev["rules_compliance_pct"] == 40.0
    title = rule_title_ru(POPULATION_RULE_ID, {"rule_type": "population_mismatch"})
    assert "ГЭРБ" in title or "возраст" in title.lower()


def test_methodist_bootstrap_requires_auto_login(feedback_env):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import rag_server

    client = TestClient(rag_server.app)
    r = client.get("/api/methodist/bootstrap")
    assert r.status_code == 404


def test_methodist_bootstrap_ok_when_auto_login(feedback_env, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import rag_server

    monkeypatch.setenv("METHODIST_UI_AUTO_LOGIN", "1")
    monkeypatch.setenv("METHODIST_REVIEWER", "Test R")
    client = TestClient(rag_server.app)
    r = client.get("/api/methodist/bootstrap")
    assert r.status_code == 200
    data = r.json()
    assert data["token"] == "test-methodist-token"
    assert data["reviewer"] == "Test R"


def test_methodist_status_includes_reviewer(feedback_env, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import rag_server

    monkeypatch.setenv("METHODIST_REVIEWER", "I.I.")
    client = TestClient(rag_server.app)
    r = client.get("/api/methodist/status")
    assert r.status_code == 200
    data = r.json()
    assert data["enabled"] is True
    assert data["default_reviewer"] == "I.I."


def test_feedback_dir_fallback_when_ml_path_not_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    fallback = tmp_path / "project" / "data" / "ml" / "feedback"
    monkeypatch.setenv("ML_FEEDBACK_DIR", "/var/data/ml/feedback")
    monkeypatch.setattr(
        "clinical_knowledge.feedback_store._DEFAULT_FEEDBACK_DIR",
        fallback,
    )
    assert feedback_dir() == fallback
    assert fallback.is_dir()


def test_build_feedback_export_tar_gz(feedback_env):
    append_feedback_event({
        "event_type": "kz_analysis",
        "analysis_id": "old",
        "text_hash": "sha256:1",
        "ts": "2026-06-01T10:00:00Z",
    })
    append_feedback_event({
        "event_type": "analysis_review",
        "analysis_id": "new",
        "text_hash": "sha256:2",
        "rating": 3,
        "verdict": "mostly_correct",
        "reviewer": "P.",
        "ts": "2026-06-15T12:00:00Z",
    })
    data, manifest = build_feedback_export_tar_gz(since="2026-06-13")
    assert manifest["event_count"] == 1
    assert manifest["files"].get("analysis_review.jsonl") == 1
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
        assert "feedback/_manifest.json" in names
        assert "feedback/analysis_review.jsonl" in names
        review_member = tar.extractfile("feedback/analysis_review.jsonl")
        assert review_member is not None
        row = json.loads(review_member.read().decode("utf-8").strip())
        assert row["analysis_id"] == "new"


def test_api_ml_feedback_export_forbidden_without_token(feedback_env):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import rag_server

    client = TestClient(rag_server.app)
    r = client.get("/api/ml/feedback/export")
    assert r.status_code == 403


def test_api_ml_feedback_export_ok_with_token(feedback_env):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import rag_server

    append_feedback_event({
        "event_type": "analysis_review",
        "analysis_id": "exp-1",
        "text_hash": "sha256:x",
        "rating": 4,
        "verdict": "mostly_correct",
        "reviewer": "P.",
    })
    client = TestClient(rag_server.app)
    headers = {"X-Methodist-Token": "test-methodist-token"}
    r = client.get("/api/ml/feedback/export", headers=headers)
    assert r.status_code == 200
    assert r.headers.get("x-feedback-event-count")
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
        assert "feedback/analysis_review.jsonl" in tar.getnames()


def test_retrieval_fix_wrong_protocol_requires_rejected_path():
    with pytest.raises(ValueError, match="rejected_path"):
        validate_and_normalize_event(
            {
                "event_type": "retrieval_fix",
                "reviewer": "test",
                "chosen_path": "minzdrav_protocols/x/right.pdf",
                "tags": ["wrong_protocol"],
            }
        )


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
