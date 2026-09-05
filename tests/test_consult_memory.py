"""Consult memory guards for Render OOM prevention."""
from __future__ import annotations

import pytest
from fastapi import HTTPException


def test_consult_forbid_full_corpus_on_render(monkeypatch):
    from clinical_knowledge.consult_memory import consult_forbid_full_corpus

    monkeypatch.delenv("CONSULT_REVIEW_FORBID_FULL_CORPUS", raising=False)
    monkeypatch.setenv("RENDER", "1")
    assert consult_forbid_full_corpus() is True


def test_cap_chunks_for_consult_limits_count_and_text():
    from clinical_knowledge.consult_memory import cap_chunks_for_consult

    chunks = [
        {"text": "x" * 5000, "chunk_id": f"c{i}", "chunk_type": "diagnostics"}
        for i in range(50)
    ]
    out = cap_chunks_for_consult(chunks)
    assert len(out) <= 24
    assert all(len(str(c.get("text") or "")) <= 2048 for c in out)


def test_consult_pipeline_skips_full_corpus_fallback(monkeypatch):
    import rag_server as rs
    from clinical_knowledge import consult_retrieval

    monkeypatch.setenv("RENDER", "1")
    monkeypatch.setenv("CONSULT_REVIEW_FORBID_FULL_CORPUS", "1")
    calls: list[list[str] | None] = []

    def _fake_retrieve(*args, **kwargs):
        calls.append(kwargs.get("path_allowlist"))
        return []

    monkeypatch.setattr(rs, "retrieve", _fake_retrieve)
    monkeypatch.setattr(rs, "_consult_render_l2_lite_enabled", lambda: False)
    monkeypatch.setattr(rs, "_consult_review_fast_mode", lambda: True)
    monkeypatch.setattr(rs, "_consult_rag_second_pass_enabled", lambda: False)
    monkeypatch.setattr(rs, "_consult_retrieve_embed_rerank", lambda: False)
    monkeypatch.setattr(rs, "get_gemini", lambda: None)
    monkeypatch.setattr(
        rs,
        "_build_consult_review_pipeline_query",
        lambda model, text: ("q", {}),
    )
    monkeypatch.setattr(
        rs,
        "_infer_icd_pipeline_from_full_query",
        lambda *a, **k: (
            {"codes_for_retrieval": ["J06.9"], "detected": [], "suggested": []},
            "q",
            "q",
            None,
            None,
        ),
    )
    monkeypatch.setattr(rs, "_merge_icd_codes_for_consult_retrieval", lambda a, t: (["J06.9"], {}))
    monkeypatch.setattr(rs, "_consult_clinical_rules_pipeline", lambda *a, **k: {})
    monkeypatch.setattr(
        consult_retrieval,
        "consult_target_protocol_paths",
        lambda **k: (["minzdrav_protocols/x/a.pdf"], {}),
    )
    monkeypatch.setattr(rs, "env_bool", lambda name, default=False: name == "CONSULT_REVIEW_STRICT_PROTOCOLS")
    monkeypatch.setattr(rs, "infer_specialties_gemini", lambda q, model: [])

    from consult_review_pipeline import iter_consult_review_pipeline

    # Именно HTTPException: pytest.raises(Exception) прошёл бы и на случайной
    # AttributeError из-за неудачного monkeypatch, то есть тест бы «зеленел»
    # не проверив strict-режим.
    with pytest.raises(HTTPException):
        for kind, payload in iter_consult_review_pipeline(
            full_text="Диагноз: J06.9 ОРВИ. Жалобы: кашель.",
            n_files=1,
            consult_docs_meta=[],
            pdf_warnings=[],
            content_signature="sig",
            category_slugs="",
        ):
            if kind == "done":
                break

    assert calls, "retrieve should have been called"
    assert all(c is not None for c in calls), "full-corpus fallback must not run on Render"
