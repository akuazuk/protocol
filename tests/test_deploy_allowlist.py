"""Dockerfile и allowlist деплоя должны описывать один и тот же набор файлов.

Деплой на GCE отправляет на VM не рабочее дерево, а `git archive` со списком
путей. Образ собирается уже на VM, из этого урезанного дерева. Значит любой
`COPY` в Dockerfile, чей источник не попал в список, ломает сборку **только на
проде**: в CI джоба `docker-images` собирает образ из полного репозитория и
такой пропуск не видит.

Так и вышло 2026-09-06: `requirements-rag.lock` появился в Dockerfile вместе с
переходом на `--require-hashes`, но в allowlist его не добавили. Релиз упал на
шаге `COPY`, прод остался на предыдущем образе. Ошибка дешёвая только потому,
что сборка идёт до подмены контейнера; при другом порядке шагов это был бы
простой.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "deploy" / "gcp-app" / "Dockerfile"
DEPLOY_SH = ROOT / "deploy" / "gcp-app" / "deploy_to_gce.sh"

# Пути, которые создаются в образе или монтируются на VM, а не приезжают
# архивом. Держим список коротким и с причиной на каждый пункт.
NOT_FROM_ARCHIVE: frozenset[str] = frozenset()


def _dockerfile_copy_sources() -> list[str]:
    """Источники всех COPY: в строке `COPY a b dst/` источники - всё кроме dst."""
    sources: list[str] = []
    for raw in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.upper().startswith("COPY "):
            continue
        if "--from=" in line:
            continue  # многостадийная сборка берёт из другого слоя, не из контекста
        parts = line.split()[1:]
        if len(parts) < 2:
            continue
        sources.extend(parts[:-1])
    return sources


def _archive_allowlist() -> list[str]:
    """Пути из `git archive ... -- <пути>` в deploy_to_gce.sh."""
    text = DEPLOY_SH.read_text(encoding="utf-8")
    entries: list[str] = []
    for block in re.finditer(r"git archive[^\n]*--[ \t]*\\\n((?:[^\n]*\\\n)*[^\n]*)", text):
        for raw in block.group(1).splitlines():
            line = raw.strip().rstrip("\\").strip()
            if not line or line.startswith("|"):
                continue
            entries.extend(line.split())
    return entries


def _covered(source: str, allowlist: list[str]) -> bool:
    """Путь покрыт, если он сам в списке или лежит под каталогом из списка."""
    src = source.rstrip("/")
    for entry in allowlist:
        allowed = entry.rstrip("/")
        if src == allowed or src.startswith(allowed + "/"):
            return True
    return False


def test_every_dockerfile_copy_source_reaches_the_vm() -> None:
    allowlist = _archive_allowlist()
    assert allowlist, "не удалось разобрать allowlist из deploy_to_gce.sh"

    missing = [
        src
        for src in _dockerfile_copy_sources()
        if src.rstrip("/") not in NOT_FROM_ARCHIVE and not _covered(src, allowlist)
    ]

    assert not missing, (
        "COPY в deploy/gcp-app/Dockerfile ссылается на пути, которые деплой не "
        f"отправляет на VM: {sorted(missing)}. Добавь их в список `git archive` в "
        "deploy/gcp-app/deploy_to_gce.sh, иначе сборка образа упадёт на проде, "
        "а не в CI."
    )


def test_lockfile_is_in_allowlist() -> None:
    """Отдельно и по имени: именно на этом файле упал релиз 2026-09-06."""
    assert _covered("requirements-rag.lock", _archive_allowlist()), (
        "requirements-rag.lock не уезжает на VM, а Dockerfile ставит зависимости "
        "через --require-hashes -r requirements-rag.lock"
    )


def test_guard_catches_a_removed_path() -> None:
    """Проверка самой проверки: без пути в списке она обязана падать."""
    allowlist = [e for e in _archive_allowlist() if e != "requirements-rag.lock"]
    assert not _covered("requirements-rag.lock", allowlist)


@pytest.mark.parametrize("path", [DOCKERFILE, DEPLOY_SH])
def test_files_exist(path: Path) -> None:
    assert path.is_file(), f"нет файла {path}"
