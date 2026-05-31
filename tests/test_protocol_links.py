"""Тесты ссылок на PDF протоколов."""
from __future__ import annotations

from clinical_knowledge.protocol_links import (
    content_disposition_inline,
    normalize_protocol_path,
    protocol_display_name,
    protocol_link_payload,
    protocol_pdf_api_path,
)


def test_protocol_pdf_api_path_valid():
    p = "minzdrav_protocols/gastroenterologiya/foo.pdf"
    url = protocol_pdf_api_path(p)
    assert url is not None
    assert url.startswith("/api/protocol-pdf?path=")
    assert "minzdrav_protocols" in url


def test_protocol_pdf_api_path_rejects_traversal():
    assert protocol_pdf_api_path("../secret.pdf") is None
    assert protocol_pdf_api_path("output/foo.pdf") is None


def test_normalize_protocol_path_adds_prefix():
    p = normalize_protocol_path("gastroenterologiya/foo.pdf")
    assert p == "minzdrav_protocols/gastroenterologiya/foo.pdf"


def test_protocol_display_name_cyrillic():
    p = "minzdrav_protocols/stomatologiya/КП1_ДНО_слюнных_желез.pdf"
    name = protocol_display_name(p)
    assert "слюнных" in name.lower()
    assert ".pdf" not in name


def test_content_disposition_ascii_safe():
    cd = content_disposition_inline("КП1_ДНО.pdf")
    assert "filename=" in cd
    assert "filename*=UTF-8" in cd


def test_protocol_link_payload():
    row = protocol_link_payload(
        "minzdrav_protocols/gastroenterologiya/foo.pdf",
        title="КП гастрит",
        matched_icd_codes=["K29"],
        icd_verified=True,
    )
    assert row is not None
    assert row["pdf_url"].startswith("/api/protocol-pdf")
    assert row["icd_verified"] is True
    assert row["matched_icd_codes"] == ["K29"]


def test_protocol_pdf_api_cyrillic_file():
    from fastapi.testclient import TestClient

    from rag_server import app

    p = "minzdrav_protocols/stomatologiya/КП1_ДНО_слюнных_желез.pdf"
    c = TestClient(app)
    r = c.get("/api/protocol-pdf", params={"path": p})
    assert r.status_code == 200
    assert "application/pdf" in (r.headers.get("content-type") or "")
