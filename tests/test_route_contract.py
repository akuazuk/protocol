"""Снимок публичного контракта маршрутов.

Зачем: `rag_server.py` - 15 тысяч строк и 149 маршрутов, и по плану
`docs/plans/2026-09-05-monolith-decomposition-v1.md` они будут вынесены в
роутеры по домену. Такой перенос обязан быть строго поведение-сохраняющим:
интерфейс врача и методиста, приложение пациента и внешние ссылки завязаны на
конкретные пути и методы.

Тест держит снимок «путь + методы» в `tests/fixtures/routes_snapshot.json`.
Любое расхождение - осознанное решение: добавили маршрут, убрали, переименовали.
Тогда снимок обновляется тем же коммитом, и изменение видно в diff, а не
всплывает в проде.

Обновить снимок:
    python3 tests/test_route_contract.py --update
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).resolve().parent / "fixtures" / "routes_snapshot.json"

# Служебные маршруты FastAPI/Starlette: они появляются от самого фреймворка и
# к контракту приложения не относятся.
IGNORED_PATHS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def current_routes() -> list[dict]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import rag_server

    out: list[dict] = []
    for route in rag_server.app.routes:
        path = getattr(route, "path", None)
        if not path or path in IGNORED_PATHS:
            continue
        methods = sorted(getattr(route, "methods", None) or [])
        # HEAD навешивается автоматически рядом с GET - в контракте не нужен.
        methods = [m for m in methods if m != "HEAD"]
        if not methods:
            # Mount статики: метод не задан, но путь важен.
            methods = ["MOUNT"]
        out.append({"path": path, "methods": methods})
    # Один путь может быть объявлен несколько раз (разные методы) - схлопываем.
    merged: dict[str, set[str]] = {}
    for item in out:
        merged.setdefault(item["path"], set()).update(item["methods"])
    return [
        {"path": p, "methods": sorted(m)} for p, m in sorted(merged.items())
    ]


def load_snapshot() -> list[dict]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_routes_match_snapshot() -> None:
    actual = current_routes()
    expected = load_snapshot()

    actual_map = {r["path"]: r["methods"] for r in actual}
    expected_map = {r["path"]: r["methods"] for r in expected}

    added = sorted(set(actual_map) - set(expected_map))
    removed = sorted(set(expected_map) - set(actual_map))
    changed = sorted(
        p
        for p in set(actual_map) & set(expected_map)
        if actual_map[p] != expected_map[p]
    )

    problems = []
    if removed:
        problems.append(
            "исчезли маршруты (интерфейс и внешние ссылки на них завязаны):\n    "
            + "\n    ".join(removed)
        )
    if added:
        problems.append("появились маршруты:\n    " + "\n    ".join(added))
    if changed:
        problems.append(
            "изменились методы:\n    "
            + "\n    ".join(
                f"{p}: {expected_map[p]} -> {actual_map[p]}" for p in changed
            )
        )

    assert not problems, (
        "Контракт маршрутов изменился.\n\n"
        + "\n\n".join(problems)
        + "\n\nЕсли изменение намеренное - обнови снимок тем же коммитом:\n"
        "    python3 tests/test_route_contract.py --update\n"
        "и убедись, что интерфейс и приложение пациента используют новые пути."
    )


def test_snapshot_is_not_empty() -> None:
    """Защита от «пустого» снимка, который проходит любой тест."""
    expected = load_snapshot()
    assert len(expected) > 100, (
        f"в снимке всего {len(expected)} маршрутов - похоже, он собран на "
        "неполностью загруженном приложении"
    )


def test_health_and_key_routes_present() -> None:
    """Точки, без которых деплой и интерфейс не работают."""
    paths = {r["path"] for r in current_routes()}
    for required in (
        "/health",
        "/health/live",
        "/api/version",
        "/api/assist",
        "/api/search/run",
        "/",
    ):
        assert required in paths, f"нет обязательного маршрута {required}"


if __name__ == "__main__":
    if "--update" in sys.argv:
        routes = current_routes()
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(
            json.dumps(routes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"снимок обновлён: {len(routes)} маршрутов -> {SNAPSHOT}")
    else:
        print("используй --update для обновления снимка")
