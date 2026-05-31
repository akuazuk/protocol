"""Тесты ссылок на PDF протоколов."""
from __future__ import annotations

from clinical_knowledge.protocol_links import protocol_display_name, protocol_pdf_api_path


def test_protocol_pdf_api_path_valid():
    p = "minzdrav_protocols/gastroenterologiya/foo.pdf"
    url = protocol_pdf_api_path(p)
    assert url is not None
    assert url.startswith("/api/protocol-pdf?path=")
    assert "minzdrav_protocols" in url


def test_protocol_pdf_api_path_rejects_traversal():
    assert protocol_pdf_api_path("../secret.pdf") is None
    assert protocol_pdf_api_path("output/foo.pdf") is None


def test_protocol_display_name():
    assert protocol_display_name("minzdrav_protocols/x/КП1.pdf") == "КП1.pdf"
