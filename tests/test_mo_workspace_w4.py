"""W4: рубрика Поиск МИС, ingest API, покрытие, без result LIKE."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "web" / "methodist" / "mis-kz-quality.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "web" / "shared" / "mo-ui.css").read_text(encoding="utf-8")
SERVER = (ROOT / "rag_server.py").read_text(encoding="utf-8")


def test_mis_page_and_nav() -> None:
    assert 'data-page="mis"' in HTML
    assert 'id="page-mis"' in HTML
    assert "Поиск МИС" in HTML
    assert 'id="mis-search-form"' in HTML
    assert 'data-mis-tab="visits"' in HTML
    assert 'data-mis-tab="labs"' in HTML
    assert 'id="mis-coverage-ring"' in HTML
    assert 'id="mis-visit-rows"' in HTML
    assert 'id="mis-lab-timeline"' in HTML
    assert "Проанализировать" in APP
    assert "function loadMisSearch" in APP
    assert "function ingestMisVisit" in APP
    assert 'page === "mis" ? "/methodist/mo/mis"' in APP
    assert ".mis-badge--in" in CSS
    assert ".lab-timeline" in CSS


def test_api_routes_and_no_result_scan() -> None:
    assert '@app.get("/api/methodist/mo/mis/visits")' in SERVER
    assert '@app.get("/api/methodist/mo/mis/labs")' in SERVER
    assert '@app.post("/api/methodist/mo/ingest-visit")' in SERVER
    assert '@app.get("/methodist/mo/mis"' in SERVER
    assert "LIKE %q%" not in SERVER
    assert "mis_protocol.result" not in SERVER
    catalog = (ROOT / "clinical_knowledge" / "mo_mis_catalog.py").read_text(encoding="utf-8")
    assert "FROM mis_data WHERE visit_id=%s LIMIT 1" in catalog
    assert "FROM mis_protocol" not in catalog
    assert "LIKE %q%" not in catalog
