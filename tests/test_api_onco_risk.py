"""Тесты /api/onco-risk: контракт ответа, аудитория, безопасность B2C."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

FORBIDDEN = ["рак", "онколог", "злокачествен", "опухол", "метастаз", "карцином"]


@pytest.fixture(scope="module")
def client():
    import rag_server as rs

    return TestClient(rs.app)


def test_onco_risk_short_text_422(client) -> None:
    r = client.post("/api/onco-risk", json={"text": ""})
    assert r.status_code == 422


def test_onco_risk_b2b_payload(client) -> None:
    r = client.post("/api/onco-risk", json={
        "text": "Жалобы: ректальное кровотечение, потеря веса, боль в животе.",
        "age": 62, "sex": "male", "labs_text": "кал на скрытую кровь положительный",
        "symptom_duration_known": True, "audience": "b2b",
    })
    assert r.status_code == 200
    data = r.json()
    a = data["assessment"]
    assert a["context"] == "symptomatic"
    assert a["triage_level"] == "refer"
    assert any(s["site"] == "colorectal" for s in a["sites"])
    site = next(s for s in a["sites"] if s["site"] == "colorectal")
    assert site["contributors"]
    assert site["p"] > 0
    assert a["sources_ru"]
    assert "не диагноз" in a["method_note_ru"]
    assert "b2c" not in data  # аудитория b2b - без пациентского блока


def test_onco_risk_b2c_only_no_numbers_no_scary(client) -> None:
    r = client.post("/api/onco-risk", json={
        "text": "Жалобы: ректальное кровотечение, потеря веса.",
        "age": 62, "sex": "male", "audience": "b2c",
    })
    assert r.status_code == 200
    data = r.json()
    assert "assessment" not in data  # без чисел для пациента
    b2c = data["b2c"]
    assert b2c["show_numbers"] is False
    assert b2c["questions"]
    for q in b2c["questions"]:
        low = q.lower()
        assert not any(w in low for w in FORBIDDEN)
        assert "%" not in q
        assert not any(ch.isdigit() for ch in q)


def test_onco_risk_both_audiences(client) -> None:
    r = client.post("/api/onco-risk", json={
        "text": "кровохарканье", "age": 70, "sex": "male", "audience": "both",
    })
    assert r.status_code == 200
    data = r.json()
    assert "assessment" in data and "b2c" in data
    assert data["server_version"]


def test_onco_risk_female_excludes_prostate(client) -> None:
    r = client.post("/api/onco-risk", json={
        "text": "учащённое мочеиспускание", "labs_text": "PSA 8",
        "age": 65, "sex": "female", "audience": "b2b",
    })
    assert r.status_code == 200
    sites = {s["site"] for s in r.json()["assessment"]["sites"]}
    assert "prostate" not in sites
