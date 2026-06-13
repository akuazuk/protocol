"""Проверка retrieve() на фикстурном чанке (без вызова внешней модели)."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def retrieve_fn():
    from rag_server import retrieve

    return retrieve


def test_retrieve_returns_something(retrieve_fn) -> None:
    out = retrieve_fn("кашель бронхит J20", max_chunks=3, max_per_path=2)
    assert isinstance(out, list)
    assert len(out) >= 1, (
        "Лексический отбор пуст - расширьте tests/fixtures/chunks.mini.jsonl или ослабьте запрос"
    )
    paths = {str(row.get("path") or "") for row in out}
    assert any("_smoke" in p or "bronch" in p.lower() or "бронх" in p.lower() for p in paths), (
        f"Ожидался чанк из mini-fixture, получено: {paths}"
    )
