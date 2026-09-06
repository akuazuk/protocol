"""Security-заголовки и CORS.

CORS и CSP читаются из окружения на импорте модуля, поэтому каждый сценарий
поднимается в отдельном процессе - иначе тесты влияли бы друг на друга через
уже импортированный `rag_server`.

Регресс, который тут закрыт (2026-09-05): на проде стояли `ALLOWED_ORIGINS=*`
и пустой `ENABLE_DEFAULT_CSP`, то есть API был открыт любому домену, а CSP
не отдавался вообще. См. docs/reports/2026-09-05-prod-state-snapshot.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Внешние источники, без которых интерфейс врача не работает.
REQUIRED_CSP_SOURCES = (
    "https://cdn.jsdelivr.net",  # Chart.js
    "https://fonts.googleapis.com",  # CSS шрифтов
    "https://fonts.gstatic.com",  # файлы шрифтов
    "blob:",  # URL.createObjectURL: превью загрузок и выгрузки
)

_PROBE = r"""
import json, os
from fastapi.testclient import TestClient
import rag_server

client = TestClient(rag_server.app)
resp = client.get("/api/version")
origin_probe = client.get(
    "/api/version", headers={"Origin": os.environ["PROBE_ORIGIN"]}
)
print("@@" + json.dumps({
    "status": resp.status_code,
    "headers": {k.lower(): v for k, v in resp.headers.items()},
    "acao": origin_probe.headers.get("access-control-allow-origin"),
    "csp_header_name": rag_server._CSP_HEADER_NAME,
}))
"""


def _probe(env: dict[str, str], probe_origin: str) -> dict:
    """Импортирует rag_server с заданным окружением и возвращает заголовки."""
    full_env = {
        **os.environ,
        "RAG_GEMINI_EMBED_RERANK": "0",
        "PROBE_ORIGIN": probe_origin,
        **env,
    }
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=ROOT,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")]
    if not marker:
        pytest.fail(
            f"проба не вернула результат\nstdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )
    return json.loads(marker[-1][2:])


def test_baseline_security_headers_present():
    result = _probe({}, "https://protocol.kravira.by")
    headers = result["headers"]
    assert result["status"] == 200
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "SAMEORIGIN"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert "max-age=" in (headers.get("strict-transport-security") or "")


def test_cors_allows_only_configured_origin():
    own = "https://protocol.kravira.by"
    allowed = _probe({"ALLOWED_ORIGINS": own}, own)
    assert allowed["acao"] == own, "собственный домен должен проходить CORS"

    rejected = _probe({"ALLOWED_ORIGINS": own}, "https://evil.example")
    assert rejected["acao"] is None, (
        "посторонний домен не должен получать Access-Control-Allow-Origin"
    )


def test_cors_closed_by_default():
    """Без ALLOWED_ORIGINS доступ только same-origin."""
    result = _probe({"ALLOWED_ORIGINS": ""}, "https://evil.example")
    assert result["acao"] is None


def test_default_csp_enabled_and_covers_ui_sources():
    result = _probe({"ENABLE_DEFAULT_CSP": "1"}, "https://protocol.kravira.by")
    csp = result["headers"].get("content-security-policy")
    assert csp, "с ENABLE_DEFAULT_CSP=1 заголовок CSP обязателен"
    assert result["csp_header_name"] == "Content-Security-Policy"

    # Иначе CSP молча ломает интерфейс врача.
    for source in REQUIRED_CSP_SOURCES:
        assert source in csp, f"CSP обязан разрешать {source}"

    # Дешёвые и безопасные ограничения: в UI нет ни <object>, ни <base>.
    assert "object-src 'none'" in csp
    assert "base-uri 'self'" in csp
    assert "frame-ancestors 'self'" in csp


def test_csp_report_only_mode_does_not_block():
    """Режим отчёта нужен для безопасного первого раската CSP."""
    result = _probe(
        {"ENABLE_DEFAULT_CSP": "1", "CSP_REPORT_ONLY": "1"},
        "https://protocol.kravira.by",
    )
    headers = result["headers"]
    assert result["csp_header_name"] == "Content-Security-Policy-Report-Only"
    assert headers.get("content-security-policy-report-only")
    assert headers.get("content-security-policy") is None


def test_csp_absent_when_disabled():
    result = _probe({"ENABLE_DEFAULT_CSP": "0"}, "https://protocol.kravira.by")
    assert result["headers"].get("content-security-policy") is None


def test_explicit_policy_overrides_default():
    custom = "default-src 'none'"
    result = _probe(
        {"ENABLE_DEFAULT_CSP": "1", "CONTENT_SECURITY_POLICY": custom},
        "https://protocol.kravira.by",
    )
    assert result["headers"].get("content-security-policy") == custom
