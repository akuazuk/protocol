"""Раздача статики по '/' отдаёт только разрешённое.

Регресс (2026-09-05): по '/' монтируется корень репозитория, а фильтр работал
как список запрещённого. Любой файл с неучтённым расширением или в неучтённом
каталоге был публичным по умолчанию - и на живом сервере отдавались с кодом 200:

    archive/docs/konkurs/06_ROI_Kravira.pdf   226 КБ  финансовая модель
    archive/docs/konkurs/03_Biznes_plan_*.pdf         бизнес-план
    deploy/gcp-app/Caddyfile, */Dockerfile            устройство инфраструктуры
    epam/scheme_mis_protocols.docx                    схема интеграции с МИС
    embeddings.json                           3.7 МБ  данные

Теперь фильтр - список разрешённого. Тесты держат обе стороны: что закрытое
закрыто и что нужное интерфейсу продолжает отдаваться (иначе «безопасную»
правку легко довести до неработающего UI).
"""

from __future__ import annotations

import subprocess

import pytest
from fastapi.testclient import TestClient

import rag_server


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(rag_server.app)


# Пути, которые публичными быть не должны. Каждый до 2026-09-05 отдавался 200.
FORBIDDEN_PATHS = [
    "/archive/docs/konkurs/06_ROI_Kravira.pdf",
    "/archive/docs/konkurs/03_Biznes_plan_Kravira_Protocol.pdf",
    "/archive/docs/konkurs/_templates/Заявка форма_1.docx",
    "/deploy/gcp-app/Caddyfile",
    "/deploy/gcp-app/Dockerfile",
    "/deploy/launchd/by.protocol.mo-daily.plist.in",
    "/epam/scheme_mis_protocols.docx",
    "/embeddings.json",
    "/Procfile",
    "/config/mo_document_kind_rules.json",
    "/ml/registry/model_manifest.json",
    "/patient-app/package.json",
    # Исходники интерфейса: те же страницы отдаются своими маршрутами,
    # раздавать их ещё и как файлы незачем.
    "/frontend/web/methodist/expert.html",
    "/frontend/web/doctor/index.html",
    # Внутренняя документация.
    "/docs/plans/README.md",
    "/docs/reports/2026-09-05-prod-state-snapshot.md",
    "/docs/deploy/gce-production-runbook.md",
    "/docs/schemas/patient_report_v2.schema.json",
    # Код, конфиги, данные, ПДн.
    "/rag_server.py",
    "/requirements.txt",
    "/.env",
    "/.git/config",
    "/AGENTS.md",
    "/data/catalog/protocol_cards.jsonl",
    "/tests/test_static_exposure.py",
]

# Пути, без которых интерфейс и презентация ломаются.
REQUIRED_PATHS = [
    "/",
    "/protocols.json",
    "/docs/mvp-presentation.html",
    "/docs/architecture-stages-print.html",
    "/docs/architecture-kravira-fhir-mis.pdf",
    "/docs/presentation-stats.json",
    # Логотипы: print-HTML ссылается относительным путём, потому что тот же
    # файл Chrome рендерит в PDF через file:// (scripts/build_architecture_pdf.py).
    "/frontend/web/shared/protocol-logo-wordmark.svg",
    # Явные маршруты общих ассетов.
    "/protocol-logo-wordmark.svg",
    "/mo-app.js",
    "/vendor/echarts.min.js",
]

# Попытки обойти фильтр.
BYPASS_PATHS = [
    "/../rag_server.py",
    "/docs/../rag_server.py",
    "/docs/../embeddings.json",
    "/docs//plans/README.md",
    "/DOCS/plans/README.md",
    "/Archive/docs/konkurs/06_ROI_Kravira.pdf",
    "/frontend/web/shared/../methodist/expert.html",
    "/frontend/web/shared/vendor/echarts.min.js",
    "/docs/%2e%2e/embeddings.json",
    "/docs%2fplans%2fREADME.md",
]


@pytest.mark.parametrize("path", FORBIDDEN_PATHS)
def test_sensitive_paths_are_not_served(client: TestClient, path: str) -> None:
    resp = client.get(path)
    assert resp.status_code in (403, 404), (
        f"{path} отдаётся публично (код {resp.status_code}). "
        "Раздача статики работает по списку разрешённого - добавлять пути туда "
        "можно только осознанно."
    )


@pytest.mark.parametrize("path", REQUIRED_PATHS)
def test_required_paths_still_served(client: TestClient, path: str) -> None:
    resp = client.get(path)
    assert resp.status_code == 200, (
        f"{path} нужен интерфейсу, но вернул {resp.status_code}. "
        "Список разрешённого сузили слишком сильно."
    )
    assert resp.content, f"{path} вернул пустое тело"


@pytest.mark.parametrize("path", BYPASS_PATHS)
def test_bypass_attempts_blocked(client: TestClient, path: str) -> None:
    resp = client.get(path)
    assert resp.status_code in (403, 404), f"{path} обошёл фильтр (код {resp.status_code})"


def test_allowlist_covers_whole_repository() -> None:
    """Ни один отслеживаемый файл вне разрешённого списка не должен быть публичным.

    Точечные пути выше проверяют известные случаи; этот тест ловит новые - файл,
    добавленный в репозиторий завтра, не станет публичным незаметно.
    """
    raw = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, check=True
    ).stdout
    tracked = [f.decode("utf-8", "replace") for f in raw.split(b"\x00") if f]
    assert tracked, "git ls-files ничего не вернул"

    allowed = [f for f in tracked if rag_server._is_allowed_static_path(f)]

    def ok(path: str) -> bool:
        if path == "protocols.json":
            return True
        if path.startswith("docs/") and path.count("/") == 1:
            return True
        return path.startswith("frontend/web/shared/") and path.rsplit(".", 1)[-1] in (
            "svg",
            "png",
        )

    unexpected = sorted(f for f in allowed if not ok(f))
    assert not unexpected, (
        "Публичными оказались файлы вне ожидаемого набора:\n  "
        + "\n  ".join(unexpected)
    )


def test_no_sensitive_extensions_are_public() -> None:
    """Расширения, которых в публичной раздаче быть не должно ни при каких путях."""
    raw = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, check=True
    ).stdout
    tracked = [f.decode("utf-8", "replace") for f in raw.split(b"\x00") if f]

    forbidden_exts = (
        ".py",
        ".sh",
        ".md",
        ".mdc",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".jsonl",
        ".csv",
        ".docx",
        ".xlsx",
        ".env",
        ".lock",
        ".in",
    )
    leaked = sorted(
        f
        for f in tracked
        if f.lower().endswith(forbidden_exts) and rag_server._is_allowed_static_path(f)
    )
    assert not leaked, "Публично отдаются файлы запрещённых типов:\n  " + "\n  ".join(
        leaked
    )
