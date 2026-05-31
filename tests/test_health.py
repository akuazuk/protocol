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
        cats = data.get("categories_top")
        assert isinstance(cats, list)
        if cats:
            assert "slug" in cats[0] and "count" in cats[0]


def test_quality_benchmark_ok(client: TestClient) -> None:
    r = client.get("/api/quality-benchmark")
    assert r.status_code == 200
    data = r.json()
    assert "pass_rate_pct" in data
    assert int(data.get("queries_total") or 0) >= 1


def test_clinical_knowledge_benchmark_ok(client: TestClient) -> None:
    r = client.get("/api/clinical-knowledge/benchmark")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert int(data.get("cases_total") or 0) >= 1


def test_clinical_knowledge_status_ok(client: TestClient) -> None:
    r = client.get("/api/clinical-knowledge/status")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert int(data.get("conditions") or 0) >= 9
    cov = data.get("rules_coverage") or {}
    assert "coverage_pct" in cov or "pdfs_total" in cov


def test_normalize_differential_field_module() -> None:
    from rag_server import normalize_differential_field

    parsed: dict = {"differential": ["а", {"text": "б"}, "  ", "c", "d", "e", "f"]}
    normalize_differential_field(parsed)
    assert parsed["differential"] == ["а", "б", "c", "d", "e"]
    normalize_differential_field(None)


def test_training_cases_ok(client: TestClient) -> None:
    r = client.get("/api/training-cases")
    assert r.status_code == 200
    assert len(r.json().get("cases") or []) >= 5


def test_pilot_analytics_demo_ok(client: TestClient) -> None:
    r = client.get("/api/pilot-analytics-demo")
    assert r.status_code == 200
    assert r.json().get("demo") is True


def test_protocol_ui_meta() -> None:
    from rag_server import protocol_ui_meta_for_path

    m = protocol_ui_meta_for_path(
        "minzdrav_protocols/revmatologiya/КП_взр_СКВ_пост_МЗ_2022_1.pdf"
    )
    assert m.get("post_mz") is True
    assert m.get("year")


def test_kz_matrix_heuristic() -> None:
    from rag_server import kz_matrix_from_clinical_detail_heuristic

    cd = {
        "extraction": {
            "diagnosis": "СКВ",
            "investigations": ["ОАК", "АНА"],
            "medications": ["гидроксихлорохин"],
            "monitoring_frequency": "каждые 3 мес",
        }
    }
    m = kz_matrix_from_clinical_detail_heuristic(
        cd, "path/x.pdf", "КП СКВ", ["M32.9"]
    )
    assert m["icd_codes"] == ["M32.9"]
    assert len(m["sections"]) >= 2
    assert m["sections"][0]["kz_section"]


def test_normalize_kz_matrix() -> None:
    from rag_server import _normalize_kz_matrix

    raw = {
        "summary_ru": "Тест",
        "sections": [
            {
                "kz_section": "Обследование",
                "items": [
                    {
                        "text": "ОАК",
                        "obligation": "required",
                        "protocol_excerpt": "…",
                    }
                ],
            }
        ],
    }
    out = _normalize_kz_matrix(raw, "p.pdf", "КП", ["M32.9"])
    assert out["sections"][0]["items"][0]["obligation"] == "required"


def test_kz_matrix_endpoint_404(client: TestClient) -> None:
    r = client.post(
        "/api/kz-matrix",
        json={"query": "тест", "path": "nonexistent.pdf", "title": ""},
    )
    assert r.status_code == 404
