"""Учётки МО Аналитики: CRUD, login, период отчётов, access reports/full."""
from pathlib import Path

from fastapi.testclient import TestClient

from clinical_knowledge import mo_app_accounts
from clinical_knowledge.mo_daily import CRM_TABLES
import rag_server

ROOT = Path(__file__).resolve().parents[1]
DOCTOR_HTML = (ROOT / "frontend" / "web" / "doctor" / "index.html").read_text(encoding="utf-8")


def _client(monkeypatch, tmp_path: Path) -> TestClient:
    from clinical_knowledge.mo_daily import initialize_warehouse

    db = tmp_path / "mo.sqlite"
    monkeypatch.setenv("METHODIST_TOKEN", "accounts-test-token")
    monkeypatch.setenv("MO_ANALYTICS_DB", str(db))
    monkeypatch.setenv("MO_DATA_ROOT", str(tmp_path))
    initialize_warehouse(db)
    mo_app_accounts.ensure_app_accounts_schema()
    return TestClient(rag_server.app)


def test_accounts_admin_crud_and_login(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    headers = {"X-Methodist-Token": "accounts-test-token"}

    denied = client.get("/api/methodist/accounts")
    assert denied.status_code == 403

    created = client.post(
        "/api/methodist/accounts",
        headers=headers,
        json={
            "login": "ivanova",
            "password": "secret-pass-1",
            "display_name": "Иванова",
            "role": "methodist",
            "mo_access": "reports",
            "reports_min_date": "2026-08-05",
        },
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["login"] == "ivanova"
    assert body["mo_access"] == "reports"
    assert body["reports_min_date"] == "2026-08-05"
    user_id = body["user_id"]

    listed = client.get("/api/methodist/accounts", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["n"] >= 1

    login = client.post(
        "/api/methodist/account/login",
        json={"login": "ivanova", "password": "secret-pass-1"},
    )
    assert login.status_code == 200
    session = login.json()["session_token"]
    sess_headers = {"X-Methodist-Session": session}

    caps = client.get("/api/methodist/mo/capabilities", headers=sess_headers)
    assert caps.status_code == 200
    payload = caps.json()
    assert payload["mo_access"] == "reports"
    assert payload["reports_min_date"] == "2026-08-05"
    assert payload["pages"]["yesterday"] is True
    assert payload["pages"]["overview"] is False

    blocked = client.get(
        "/api/methodist/mo/daily-report?date=2026-08-01",
        headers=sess_headers,
    )
    assert blocked.status_code == 403

    # Full BI path blocked for reports access.
    month = client.get("/api/methodist/mo/month-report?period=month", headers=sess_headers)
    assert month.status_code == 403

    patched = client.patch(
        f"/api/methodist/accounts/{user_id}",
        headers=headers,
        json={
            "login": "ivanova",
            "mo_access": "full",
            "role": "methodist",
            "reports_min_date": "2026-08-01",
            "active": True,
        },
    )
    assert patched.status_code == 200
    assert patched.json()["mo_access"] == "full"


def test_crm_tables_include_app_accounts() -> None:
    assert "crm_app_user" in CRM_TABLES
    assert "crm_app_session" in CRM_TABLES


def test_methodist_tab_is_accounts_admin_not_summary_stub() -> None:
    assert "МО Аналитика - учётные записи" in DOCTOR_HTML
    assert "mo-account-create" in DOCTOR_HTML
    assert "mo-accounts-table" in DOCTOR_HTML
    assert "Массовый анализ медицинских записей" not in DOCTOR_HTML
    assert "CRM/BI-дашборд МО" not in DOCTOR_HTML
    assert "function createMethodistMoAccount()" in DOCTOR_HTML
