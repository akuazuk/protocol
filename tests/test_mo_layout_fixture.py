from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend/web/methodist/mo-layout-fixture.html").read_text(encoding="utf-8")
JS = (ROOT / "frontend/web/methodist/mo-layout-fixture.js").read_text(encoding="utf-8")


def test_fixture_route_does_not_call_the_warehouse():
    from fastapi.testclient import TestClient

    import rag_server

    client = TestClient(rag_server.app)
    page = client.get("/methodist/mo-layout-fixture.html")
    script = client.get("/methodist/mo-layout-fixture.js")
    assert page.status_code == 200
    assert script.status_code == 200
    assert "fx-critical" in page.text
    assert "/api/" not in page.text
    assert "mo-app.js" not in page.text


def test_fixture_is_static_and_has_five_visits():
    assert "mo-app.js" not in HTML
    assert "/api/" not in HTML
    assert "fetch(" not in JS
    assert "XMLHttpRequest" not in JS
    for key in ("fx-major", "fx-nsaid", "fx-empty-dx", "fx-stale", "fx-critical"):
        assert key in HTML
        assert key in JS
    assert "Текст осмотра фикстуры" in HTML
    assert "Major: Препарат А + Препарат Б" in HTML
    assert "Одновременно два НПВП" in HTML
    assert "Оценка старше замечания" in HTML
    assert "mo-ui.css" in HTML
    assert 'data-overall-grade="critical"' in HTML
