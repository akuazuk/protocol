"""Извлечение текста КЗ из PDF и текстовых форматов."""
from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient


def test_decode_utf8_text() -> None:
    import rag_server as rs

    txt, warns = rs.extract_consult_text_from_bytes(
        "Консультативное заключение\nДиагноз: J20".encode("utf-8"),
        "kz.txt",
    )
    assert "J20" in txt
    assert not warns


def test_decode_cp1251_text() -> None:
    import rag_server as rs

    txt, _ = rs.extract_consult_text_from_bytes(
        "Жалобы пациента".encode("cp1251"),
        "legacy.txt",
    )
    assert "Жалобы" in txt


def test_extract_docx_minimal() -> None:
    import rag_server as rs

    doc_xml = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:body>"
        b"<w:p><w:r><w:t>Diag section</w:t></w:r></w:p>"
        b"<w:p><w:r><w:t>MKB J20.9</w:t></w:r></w:p>"
        b"</w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", doc_xml)
    txt, _ = rs.extract_consult_text_from_bytes(buf.getvalue(), "kz.docx")
    assert "J20.9" in txt
    assert "Diag section" in txt


def test_extract_rtf_basic() -> None:
    import rag_server as rs

    raw = r"{\rtf1\ansi Текст заключения\par J20.9}".encode("utf-8")
    txt, _ = rs.extract_consult_text_from_bytes(raw, "kz.rtf")
    assert "J20.9" in txt


def test_rejects_unknown_extension() -> None:
    import rag_server as rs
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        rs.extract_consult_text_from_bytes(b"data", "file.exe")
    assert exc.value.status_code == 400


@pytest.fixture(scope="module")
def client() -> TestClient:
    import rag_server as rs

    return TestClient(rs.app)


def test_consult_review_accepts_txt(client, monkeypatch) -> None:
    import rag_server as rs

    rs._consult_review_cache.clear()
    rs._consult_cache_order.clear()
    monkeypatch.setenv("CONSULT_REVIEW_CACHE", "0")
    monkeypatch.setenv("CONSULT_REVIEW_STRICT_PROTOCOLS", "0")

    called = {"n": 0}

    def fake_extract(data: bytes, filename: str = "") -> tuple[str, list[str]]:
        called["n"] += 1
        return ("текст заключения для проверки", [])

    monkeypatch.setattr(rs, "extract_consult_text_from_bytes", fake_extract)
    monkeypatch.setattr(rs, "get_gemini", lambda: object())
    monkeypatch.setattr(
        rs,
        "_build_consult_review_pipeline_query",
        lambda model, t: ("=== Жалобы ===\n\nтекст", {"focus_source": "test"}),
    )
    monkeypatch.setattr(
        rs,
        "_infer_icd_pipeline_from_full_query",
        lambda q, model: ({"codes_for_retrieval": [], "detected": [], "suggested": []}, "q", "q", None, None),
    )
    monkeypatch.setattr(rs, "_merge_icd_codes_for_consult_retrieval", lambda a, t: ([], {}))
    monkeypatch.setattr(rs, "infer_specialties_gemini", lambda q, model: [])
    monkeypatch.setattr(rs, "consult_demographics_banner_from_kz", lambda t: ("", {}))
    monkeypatch.setattr(rs, "_consult_icd_banner_for_retrieval", lambda d, m: "")
    fake_rows = [
        {
            "path": "fake/p.pdf",
            "kind": "treatment",
            "excerpt": "текст протокола",
            "score": 0.9,
        }
    ]
    monkeypatch.setattr(rs, "retrieve", lambda *a, **k: list(fake_rows))
    monkeypatch.setattr(rs, "filter_retrieval_by_audience", lambda rows, q, routing: (rows, None, False))
    monkeypatch.setattr(rs, "_consult_needles_icd_fragments_consult_review", lambda d, m: [])
    monkeypatch.setattr(rs, "_consult_sort_retrieval_by_icd_fragments_first", lambda rows, n: rows)
    monkeypatch.setattr(
        rs, "_consult_precise_links_for_icd_in_fragments", lambda rows, diag_block_icd, merged_icd: ([], "")
    )
    monkeypatch.setattr(rs, "_build_review_chunks_context", lambda rows, mx: ("ctx", ["fake/p.pdf"]))
    monkeypatch.setattr(rs, "_consult_review_paths_hint", lambda paths, retrieved, icd_needles: "hint")
    monkeypatch.setattr(rs, "_consult_ui_protocol_fragments", lambda rows, paths: [])
    monkeypatch.setattr(rs, "_consult_oncology_flags", lambda frags, t: {"any": False})
    monkeypatch.setattr(rs, "_icd_client_payload", lambda a: {})
    monkeypatch.setattr(
        rs,
        "_consult_review_synthesize",
        lambda *a, **k: {"overall_compliance_pct": 80, "criteria": [], "summary_ru": "ok"},
    )

    r = client.post(
        "/api/consult-review",
        files={"files": ("test_kz.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 200, r.text
    assert called["n"] == 1
