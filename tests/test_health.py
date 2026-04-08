"""Смоук: приложение поднимается с фикстурным корпусом, /health отвечает."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from rag_server import app

    return TestClient(app)


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert int(data.get("chunks") or 0) >= 1


def test_specialties_ok(client: TestClient) -> None:
    r = client.get("/api/specialties")
    assert r.status_code == 200
    data = r.json()
    assert "specialties" in data
    assert isinstance(data["specialties"], list)
    assert len(data["specialties"]) >= 1
