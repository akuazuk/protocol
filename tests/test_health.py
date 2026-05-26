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


def test_corpus_stats_ok(client: TestClient) -> None:
    r = client.get("/api/corpus-stats")
    assert r.status_code == 200
    data = r.json()
    assert data.get("specialties_catalog") == 24
    if data.get("index_csv_available"):
        assert int(data.get("protocols_in_index") or 0) >= 1


def test_quality_benchmark_ok(client: TestClient) -> None:
    r = client.get("/api/quality-benchmark")
    assert r.status_code == 200
    data = r.json()
    assert "pass_rate_pct" in data
    assert int(data.get("queries_total") or 0) >= 1
