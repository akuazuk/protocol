"""Lite /api/assist: быстрый JSON только с protocols (фаза A1+B1)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import rag_server as rs

    return TestClient(rs.app)


class _FakeResp:
    def __init__(self, text: str) -> None:
        self._text = text
        self.prompt_feedback = None

        class _Cand:
            finish_reason = 1

        self.candidates = [_Cand()]

    @property
    def text(self) -> str:
        return self._text


def _assist_mocks(monkeypatch: pytest.MonkeyPatch, *, answer_json: str) -> list[str]:
    import rag_server as rs

    prompts: list[str] = []
    icd_analysis = {"codes_for_retrieval": ["K64.9"], "detected": [], "suggested": []}
    fake_rows = [
        {
            "path": "minzdrav_protocols/khirurgiya/hemorrhoid.pdf",
            "kind": "treatment",
            "excerpt": "Лечение геморроя.",
            "score": 0.9,
            "lexical_score": 0.8,
            "routing_multiplier": 1.0,
        }
    ]

    monkeypatch.setenv("RAG_ASSIST_LITE", "1")
    monkeypatch.setenv("RAG_ICD_FAST_AUTO", "0")
    monkeypatch.setattr(rs, "get_gemini", lambda: object())
    monkeypatch.setattr(
        rs,
        "_infer_icd_pipeline_from_full_query",
        lambda query, model, **kwargs: (icd_analysis, query, query, None, None),
    )

    def _no_specialty(q, model):
        raise AssertionError("infer_specialties_gemini should be skipped when ICD present")

    monkeypatch.setattr(rs, "infer_specialties_gemini", _no_specialty)
    monkeypatch.setattr(rs, "retrieve", lambda *a, **k: list(fake_rows))
    monkeypatch.setattr(
        rs, "filter_retrieval_by_audience", lambda rows, q, routing: (rows, None, False)
    )
    monkeypatch.setattr(
        rs, "maybe_refine_icd_with_gemini_after_retrieve", lambda *a, **k: None
    )
    monkeypatch.setattr(rs, "_try_icd_fast_assist", lambda *a, **k: None)

    def _capture_prompt(model, prompt):
        prompts.append(prompt)
        return _FakeResp(answer_json)

    monkeypatch.setattr(rs, "generate_gemini", _capture_prompt)
    return prompts


def test_assist_lite_default_strips_verbose_fields(client, monkeypatch) -> None:
    answer = (
        '{"protocols":[{"path":"minzdrav_protocols/khirurgiya/hemorrhoid.pdf",'
        '"title":"Геморрой","match_reason":"Подходит.","confidence_score":0.9}],'
        '"summary":"should strip","differential":["x"],"questions_for_patient":["q"]}'
    )
    prompts = _assist_mocks(monkeypatch, answer_json=answer)
    r = client.post("/api/assist", json={"query": "геморрой K64.9"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("assist_lite") is True
    j = data.get("llm_json") or {}
    assert "summary" not in j
    assert "differential" not in j
    assert "questions_for_patient" not in j
    assert prompts
    assert "SYSTEM_JSON_LITE" not in prompts[0]  # constant name not in prompt
    assert "summary" not in prompts[0].split("Схема")[0] or "НЕ добавляй summary" in prompts[0]


def test_assist_full_mode_uses_full_prompt(client, monkeypatch) -> None:
    import rag_server as rs

    answer = '{"protocols":[],"summary":"ok","differential":[],"questions_for_patient":[]}'
    prompts = _assist_mocks(monkeypatch, answer_json=answer)
    r = client.post(
        "/api/assist",
        json={"query": "геморрой K64.9", "assist_full": True},
    )
    assert r.status_code == 200, r.text
    assert r.json().get("assist_lite") is False
    assert prompts
    assert "differential" in prompts[0]
