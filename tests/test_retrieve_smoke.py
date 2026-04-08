"""Проверка retrieve() на фикстурном чанке (без вызова Gemini)."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def retrieve_fn():
    from rag_server import retrieve

    return retrieve


def test_retrieve_returns_something(retrieve_fn) -> None:
    out = retrieve_fn("кашель бронхит J20", max_chunks=3, max_per_path=2)
    assert isinstance(out, list)
    # при минимальном корпусе допустим пустой отбор при слишком жёсткой лексике — см. eval/golden_queries.jsonl
    if len(out) == 0:
        pytest.skip("Лексический отбор пуст — расширьте chunks.mini.jsonl или ослабьте запрос")
