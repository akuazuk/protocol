"""Тесты ссылок на PDF протоколов."""
from __future__ import annotations

from clinical_knowledge.protocol_links import (
    beautify_protocol_title,
    content_disposition_inline,
    dedupe_protocol_rows,
    normalize_protocol_path,
    protocol_display_name,
    protocol_link_payload,
    protocol_nav_api_path,
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
    assert "_" not in name


def test_beautify_protocol_title_long_filename():
    raw = (
        "КП_Диагностика_и_лечение_пациентов_(взрослое_население)_"
        "с_тромбозом_глубоких_вен_пост_МЗ_от_22_03_2022_№17"
    )
    name = beautify_protocol_title(raw)
    assert "_" not in name
    assert "взрослое население" in name
    assert "22.03.2022" in name
    assert "№17" in name
    assert "тромбозом" in name.lower()


def test_protocol_display_name_uses_registry_with_underscores():
    p = "minzdrav_protocols/bolezni-sistemy-krovoobrashcheniya/foo.pdf"
    raw = "КП_Диагностика_и_лечение_пациентов_(взрослое_население)_с_тромбозом_глубоких_вен_пост_МЗ_от_22_03_2022_№17"
    name = protocol_display_name(p, registry_title=raw)
    assert "_" not in name
    assert "22.03.2022" in name


def test_protocol_display_name_skips_amendment_boilerplate():
    from clinical_knowledge.protocol_links import title_looks_truncated

    garbage = (
        "клинического протокола» заменить словами «в соответствии с клиническим протоколом"
    )
    assert title_looks_truncated(garbage)
    path = (
        "minzdrav_protocols/allergologiya-immunologiya/"
        "КП_Диагностика_и_лечение_пациентов_д-нас_с_бронхиальной_астмой_пост_МЗ_2025_38.pdf"
    )
    name = protocol_display_name(
        path, registry_title=garbage, prefer_filename_if_truncated=True
    )
    assert "астмой" in name.lower()
    assert "заменить" not in name.lower()


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
    assert row["nav_url"].startswith("/proto-viewer.html?path=")
    assert row["url"] == row["nav_url"]
    assert row["icd_verified"] is True
    assert row["matched_icd_codes"] == ["K29"]


def test_protocol_nav_api_path():
    url = protocol_nav_api_path("minzdrav_protocols/gastroenterologiya/foo.pdf", section="treatment")
    assert url is not None
    assert url.startswith("/proto-viewer.html?path=")
    assert "section=treatment" in url


def test_dedupe_protocol_rows_by_basename_and_title():
    rows = [
        {
            "path": "minzdrav_protocols/akusherstvo-ginekologiya/КП_вены.pdf",
            "title": "Диагностика и лечение ХЗВ",
            "confidence_score": 0.7,
        },
        {
            "path": "minzdrav_protocols/bolezni-sistemy-krovoobrashcheniya/КП_вены.pdf",
            "title": "Диагностика и лечение ХЗВ",
            "confidence_score": 0.9,
        },
        {
            "path": "minzdrav_protocols/khirurgiya/other.pdf",
            "title": "Другой протокол",
            "confidence_score": 0.5,
        },
    ]
    out = dedupe_protocol_rows(rows)
    assert len(out) == 2
    assert out[0]["path"].endswith("КП_вены.pdf")
    assert "bolezni-sistemy-krovoobrashcheniya" in out[0]["path"]


def test_protocol_pdf_api_cyrillic_file(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import rag_server as rs

    rel = "minzdrav_protocols/stomatologiya/КП1_ДНО_слюнных_желез.pdf"
    pdf_dir = tmp_path / "minzdrav_protocols" / "stomatologiya"
    pdf_dir.mkdir(parents=True)
    pdf_path = pdf_dir / "КП1_ДНО_слюнных_желез.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test fixture\n")
    monkeypatch.setattr(rs, "ROOT", tmp_path)

    c = TestClient(rs.app)
    r = c.get("/api/protocol-pdf", params={"path": rel})
    assert r.status_code == 200
    assert "application/pdf" in (r.headers.get("content-type") or "")
