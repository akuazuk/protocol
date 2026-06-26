"""Извлечение текста КЗ из PDF и текстовых форматов."""
from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient


def test_jpg_extension_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_server as rs

    monkeypatch.setattr(
        "clinical_knowledge.image_ocr.ocr_image_bytes",
        lambda data: ("Консультативное заключение\nДиагноз I10", ["Текст извлечён из фото через OCR"]),
    )
    # minimal JPEG header
    jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    txt, warns = rs.extract_consult_text_from_bytes(jpeg, "photo.jpg")
    assert "I10" in txt
    assert any("OCR" in w for w in warns)


def test_jpg_extension_rejected_before_fix_was_plain() -> None:
    import rag_server as rs

    assert ".jpg" in rs.CONSULT_REVIEW_ALLOWED_EXTENSIONS
    assert ".jpeg" in rs.CONSULT_REVIEW_ALLOWED_EXTENSIONS


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


def test_pdf_with_bom_prefix() -> None:
    import rag_server as rs

    pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n"
    # minimal pdf won't extract text but should not reject signature
    try:
        from pypdf import PdfWriter

        buf = io.BytesIO()
        w = PdfWriter()
        w.add_blank_page(width=72, height=72)
        w.write(buf)
        pdf = buf.getvalue()
    except ImportError:
        pytest.skip("pypdf required")

    txt, warns = rs.extract_consult_text_from_bytes(b"\xef\xbb\xbf" + pdf, "pl_new.pdf")
    assert isinstance(txt, str)
    assert not any("сигнатур" in w.lower() for w in warns)


def test_pdf_mislabeled_docx_fallback() -> None:
    import rag_server as rs

    doc_xml = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:body><w:p><w:r><w:t>Diag J20.9</w:t></w:r></w:p></w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", doc_xml)
    txt, _ = rs.extract_consult_text_from_bytes(buf.getvalue(), "pl_new.pdf")
    assert "J20.9" in txt


def test_pdf_docx_with_bom_prefix() -> None:
    import rag_server as rs

    doc_xml = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:body><w:p><w:r><w:t>Diag M54.1</w:t></w:r></w:p></w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", doc_xml)
    payload = b"\xef\xbb\xbf" + buf.getvalue()
    txt, _ = rs.extract_consult_text_from_bytes(payload, "report_g_1.pdf")
    assert "M54.1" in txt


def test_pypdf_empty_pymupdf_fallback(monkeypatch) -> None:
    import rag_server as rs
    from clinical_knowledge import text_extract as te

    def fake_pypdf(_data: bytes, *, max_pages: int = 200):
        return "", [], None

    def fake_mupdf(_data: bytes, *, max_pages: int = 200):
        return "Текст заключения M54.1", []

    monkeypatch.setattr(te, "extract_pdf_text_pypdf", fake_pypdf)
    monkeypatch.setattr(te, "extract_pdf_text_pymupdf", fake_mupdf)
    txt, warns = rs.extract_pdf_text_from_bytes(b"%PDF-1.4 fake")
    assert "M54.1" in txt
    assert any("PyMuPDF" in w for w in warns)


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
