"""P0 auth unify: shared-token SSO markers + expert route redirect."""
from pathlib import Path

from fastapi.testclient import TestClient

import rag_server

ROOT = Path(__file__).resolve().parents[1]
DOCTOR_HTML = (ROOT / "frontend" / "web" / "doctor" / "index.html").read_text(encoding="utf-8")
MO_API = (ROOT / "frontend" / "web" / "shared" / "mo-api.js").read_text(encoding="utf-8")
MO_APP = (ROOT / "frontend" / "web" / "shared" / "mo-app.js").read_text(encoding="utf-8")


def test_expert_routes_redirect_to_methodist_mo() -> None:
    client = TestClient(rag_server.app)
    for path in (
        "/methodist/expert",
        "/methodist/expert.html",
        "/methodist/expert/yesterday",
        "/methodist/expert/reports",
    ):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 302, path
        assert response.headers["location"] == "/methodist/mo", path


def test_methodist_cabinet_persists_token_to_local_storage() -> None:
    assert "function writeMethodistStorage(key, value)" in DOCTOR_HTML
    assert "function readMethodistStorage(key)" in DOCTOR_HTML
    assert "function clearMethodistStorageCreds()" in DOCTOR_HTML
    assert "writeMethodistStorage(METHODIST_STORAGE_TOKEN, tok)" in DOCTOR_HTML
    assert "localStorage.setItem(key, next)" in DOCTOR_HTML
    assert "sessionStorage.getItem(key) || localStorage.getItem(key)" in DOCTOR_HTML


def test_mo_api_syncs_token_across_storages_and_prefers_methodist_on_mo() -> None:
    assert "function setToken(value)" in MO_API
    assert "function clearToken()" in MO_API
    assert "sessionStorage.getItem(key) || localStorage.getItem(key)" in MO_API
    assert "prefer methodist token" in MO_API
    assert "isExpertAudience()" in MO_API


def test_mo_app_keeps_session_on_permission_403() -> None:
    assert "function shouldForceReauth(status, detail)" in MO_APP
    assert "async function handleHttpAuth(response)" in MO_APP
    assert "await handleHttpAuth(response)" in MO_APP
    assert 'text.indexOf("роль") >= 0' in MO_APP
