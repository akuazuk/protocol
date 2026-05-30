"""Тесты /api/assist: валидация, путь без ключа модели и успешный путь с мок-моделью.

Внешняя модель не вызывается — все точки обращения к ней подменяются monkeypatch.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import rag_server as rs

    return TestClient(rs.app)


def test_assist_short_query_422(client) -> None:
    r = client.post("/api/assist", json={"query": "a"})
    assert r.status_code == 422


def test_assist_no_model_key_503(client, monkeypatch) -> None:
    """Без ключа API get_gemini() должен вернуть 503 (а не 500)."""
    import rag_server as rs

    monkeypatch.setattr(rs, "_model", None, raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = client.post("/api/assist", json={"query": "кашель и температура"})
    assert r.status_code == 503


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


def test_assist_success_mocked(client, monkeypatch) -> None:
    """Успешный путь: модель и тяжёлые шаги подменены — проверяем сборку ответа."""
    import rag_server as rs

    icd_analysis = {"codes_for_retrieval": ["J20"], "detected": [], "suggested": []}

    monkeypatch.setattr(rs, "get_gemini", lambda: object())
    monkeypatch.setattr(
        rs,
        "_infer_icd_pipeline_from_full_query",
        lambda query, model: (icd_analysis, query, query, None, None),
    )
    monkeypatch.setattr(rs, "infer_specialties_gemini", lambda q, model: [])
    fake_rows = [
        {
            "path": "fake/protocol.pdf",
            "kind": "treatment",
            "excerpt": "Лечение бронхита: бронходилататоры.",
            "score": 0.9,
            "lexical_score": 0.8,
            "routing_multiplier": 1.0,
        }
    ]
    monkeypatch.setattr(rs, "retrieve", lambda *a, **k: list(fake_rows))
    monkeypatch.setattr(
        rs, "filter_retrieval_by_audience", lambda rows, q, routing: (rows, None, False)
    )
    monkeypatch.setattr(
        rs, "maybe_refine_icd_with_gemini_after_retrieve", lambda *a, **k: None
    )
    answer_json = (
        '{"answer_markdown": "Краткий ответ по протоколу.", '
        '"protocols": [{"path": "fake/protocol.pdf", "title": "Бронхит", '
        '"confidence_score": "medium", "rag_support": 0.5}], '
        '"differential": []}'
    )
    monkeypatch.setattr(rs, "generate_gemini", lambda model, prompt: _FakeResp(answer_json))

    r = client.post(
        "/api/assist",
        json={"query": "кашель бронхит", "inline_clinical_detail": False},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "llm_text" in data
    assert "llm_json" in data
    assert "retrieval_embedding" in data
    assert isinstance(data.get("llm_json"), dict)
