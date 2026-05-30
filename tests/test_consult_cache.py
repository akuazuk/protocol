"""Воспроизводимость проверки КЗ: один и тот же PDF -> идентичный результат (кэш по контент-хэшу).

Гарантирует требование пользователя: повторная загрузка того же файла даёт одинаковое
«Ориентировочное соответствие», даже если модель недетерминирована.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import rag_server as rs

    return TestClient(rs.app)


def test_cache_key_stable_and_sensitive() -> None:
    import rag_server as rs

    # Ключ по нормализованному содержанию: разный регистр/пробелы дают тот же ключ.
    k1 = rs._consult_cache_key(rs._normalize_for_cache("Текст  Заключения"), "pulmonologiya")
    k2 = rs._consult_cache_key(rs._normalize_for_cache("текст заключения"), "pulmonologiya")
    k3 = rs._consult_cache_key(rs._normalize_for_cache("другой текст"), "pulmonologiya")
    assert k1 == k2
    assert k1 != k3


def test_cache_put_get_roundtrip() -> None:
    import rag_server as rs

    rs._consult_review_cache.clear()
    rs._consult_cache_order.clear()
    key = "k-test-1"
    rs._consult_cache_put(key, {"ok": True, "review": {"overall_compliance_pct": 73}})
    got = rs._consult_cache_get(key)
    assert got is not None
    assert got["cached_result"] is True
    assert got["review"]["overall_compliance_pct"] == 73
    # Изменение возвращённой копии не портит кэш
    got["review"]["overall_compliance_pct"] = 0
    again = rs._consult_cache_get(key)
    assert again["review"]["overall_compliance_pct"] == 73


def test_same_pdf_returns_identical_result(client, monkeypatch) -> None:
    import rag_server as rs

    rs._consult_review_cache.clear()
    rs._consult_cache_order.clear()
    monkeypatch.setenv("CONSULT_REVIEW_CACHE", "1")
    monkeypatch.setenv("CONSULT_REVIEW_RAG_SECOND_PASS", "0")

    icd_analysis = {"codes_for_retrieval": [], "detected": [], "suggested": []}
    fake_rows = [
        {
            "path": "fake/p.pdf",
            "kind": "treatment",
            "excerpt": "текст протокола",
            "score": 0.9,
            "lexical_score": 0.8,
            "routing_multiplier": 1.0,
        }
    ]

    monkeypatch.setattr(rs, "extract_pdf_text_from_bytes", lambda data: ("текст заключения пациента", []))
    monkeypatch.setattr(rs, "get_gemini", lambda: object())
    monkeypatch.setattr(rs, "_build_consult_review_pipeline_query", lambda model, t: ("=== Жалобы ===\n\nтекст", {"focus_source": "test"}))
    monkeypatch.setattr(rs, "_infer_icd_pipeline_from_full_query", lambda q, model: (icd_analysis, "q", "q_rag", None, None))
    monkeypatch.setattr(rs, "_merge_icd_codes_for_consult_retrieval", lambda a, t: ([], {"diag_block_icd_codes": []}))
    monkeypatch.setattr(rs, "infer_specialties_gemini", lambda q, model: [])
    monkeypatch.setattr(rs, "consult_demographics_banner_from_kz", lambda t: ("", {}))
    monkeypatch.setattr(rs, "_consult_icd_banner_for_retrieval", lambda d, m: "")
    monkeypatch.setattr(rs, "retrieve", lambda *a, **k: list(fake_rows))
    monkeypatch.setattr(rs, "filter_retrieval_by_audience", lambda rows, q, routing: (rows, None, False))
    monkeypatch.setattr(rs, "_consult_needles_icd_fragments_consult_review", lambda d, m: [])
    monkeypatch.setattr(rs, "_consult_sort_retrieval_by_icd_fragments_first", lambda rows, n: rows)
    monkeypatch.setattr(rs, "_consult_precise_links_for_icd_in_fragments", lambda rows, diag_block_icd, merged_icd: ([], ""))
    monkeypatch.setattr(rs, "_build_review_chunks_context", lambda rows, mx: ("ctx", ["fake/p.pdf"]))
    monkeypatch.setattr(rs, "_consult_review_paths_hint", lambda paths, retrieved, icd_needles: "hint")
    monkeypatch.setattr(rs, "_consult_ui_protocol_fragments", lambda rows, paths: [])
    monkeypatch.setattr(rs, "_consult_oncology_flags", lambda frags, t: {"any": False})
    monkeypatch.setattr(rs, "_icd_client_payload", lambda a: {})

    calls = {"n": 0}

    def fake_synth(model, consult_excerpt, protocol_ctx, paths_hint, extra_context=""):
        calls["n"] += 1
        # имитируем недетерминированную модель: каждый вызов даёт разный %
        return {"overall_compliance_pct": 50 + calls["n"], "criteria": [], "summary_ru": "x"}

    monkeypatch.setattr(rs, "_consult_review_synthesize", fake_synth)

    pdf_bytes = b"%PDF-1.4 fake consult content"
    files = {"files": ("pl_2_d_s.pdf", pdf_bytes, "application/pdf")}

    r1 = client.post("/api/consult-review", files=files)
    assert r1.status_code == 200, r1.text
    d1 = r1.json()

    files2 = {"files": ("pl_2_d_s.pdf", pdf_bytes, "application/pdf")}
    r2 = client.post("/api/consult-review", files=files2)
    assert r2.status_code == 200, r2.text
    d2 = r2.json()

    # Тяжёлый синтез вызван ровно один раз — второй ответ из кэша
    assert calls["n"] == 1
    assert d1["review"]["overall_compliance_pct"] == d2["review"]["overall_compliance_pct"]
    assert d1.get("cached_result") is False
    assert d2.get("cached_result") is True
