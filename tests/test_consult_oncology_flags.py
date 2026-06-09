"""Тесты онко-эвристики: kz_only не даёт баннер из каталога novoobrazovaniya."""
from __future__ import annotations

import rag_server as rs


def test_oncology_kz_only_ignores_novoobrazovaniya_catalog(monkeypatch):
    monkeypatch.setenv("CONSULT_ONCO_SCAN_SOURCE", "kz_only")
    frags = [
        {
            "path": "minzdrav_protocols/novoobrazovaniya/kp_test.pdf",
            "title": "КП ЗНО",
        }
    ]
    text = "Диагноз: I80.1 Флеботромбоз. Рекомендации: ривароксабан."
    flags = rs._consult_oncology_flags(frags, text)
    assert flags["scan_source"] == "kz_only"
    assert flags["protocol_hit"] is True
    assert flags["consultation_hit"] is False
    assert flags["any"] is False
    assert not flags.get("banner_ru")


def test_oncology_legacy_shows_catalog_banner(monkeypatch):
    monkeypatch.setenv("CONSULT_ONCO_SCAN_SOURCE", "legacy")
    frags = [
        {
            "path": "minzdrav_protocols/novoobrazovaniya/kp_test.pdf",
            "title": "КП ЗНО",
        }
    ]
    text = "Диагноз: I80.1 Флеботромбоз."
    flags = rs._consult_oncology_flags(frags, text)
    assert flags["any"] is True
    assert "новообразования" in (flags.get("banner_ru") or "")


def test_oncology_kz_only_consultation_hit(monkeypatch):
    monkeypatch.setenv("CONSULT_ONCO_SCAN_SOURCE", "kz_only")
    flags = rs._consult_oncology_flags([], "Подозрение на злокачественное новообразование.")
    assert flags["consultation_hit"] is True
    assert flags["any"] is True
