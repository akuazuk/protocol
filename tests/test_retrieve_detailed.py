"""Детальные проверки retrieve() на мини-корпусе (conftest)."""
from __future__ import annotations

import pytest

from eval.retrieval_checks import (
    check_expected_paths,
    check_must_substrings,
    validate_retrieval_schema,
)


@pytest.fixture(scope="module")
def retrieve_fn():
    from rag_server import retrieve

    return retrieve


def test_retrieve_schema_and_scores(retrieve_fn) -> None:
    out = retrieve_fn("кашель бронхит J20.9", max_chunks=4, max_per_path=2)
    assert out
    warn = validate_retrieval_schema(out)
    assert not warn, warn
    assert all(float(r.get("score") or 0) >= 0 for r in out)


def test_retrieve_bronchitis_path(retrieve_fn) -> None:
    out = retrieve_fn("острый бронхит кашель", max_chunks=3, max_per_path=2)
    assert out
    ok, _ = check_expected_paths(out, ["test_protocol_smoke"])
    assert ok


def test_retrieve_second_protocol_ambulatory(retrieve_fn) -> None:
    out = retrieve_fn(
        "клинический протокол амбулаторного лечения",
        max_chunks=6,
        max_per_path=2,
    )
    assert out
    ok, _ = check_expected_paths(out, ["general_ambulatory_mini"])
    assert ok
    m_ok, miss = check_must_substrings(out, ["протокол"])
    assert m_ok, miss


def test_retrieve_nonsense_often_empty(retrieve_fn) -> None:
    out = retrieve_fn("zzzz_nonexistent_token_xyz_12345", max_chunks=4, max_per_path=2)
    assert out == []


def test_retrieve_icd_code_in_query_without_icd_pipeline(retrieve_fn) -> None:
    """Код МКБ в тексте запроса даёт лекс-токены (extract_icd_codes_raw + icd_tokens_for_lex)."""
    out = retrieve_fn("J20.9", max_chunks=4, max_per_path=2)
    assert out
    assert any("j20" in str(r.get("excerpt", "")).lower() for r in out) or any(
        "test_protocol" in str(r.get("path", "")).lower() for r in out
    )


def test_retrieve_top_path_stable(retrieve_fn) -> None:
    out = retrieve_fn("J20.9 бронхит кашель", max_chunks=2, max_per_path=1)
    assert out
    assert "test_protocol_smoke" in (out[0].get("path") or "")


def test_retrieve_forbidden_not_in_top(retrieve_fn) -> None:
    from eval.retrieval_checks import check_forbidden_substrings

    out = retrieve_fn("кашель бронхит", max_chunks=6, max_per_path=3)
    assert out
    ok, found = check_forbidden_substrings(out, ["general_ambulatory"])
    assert ok and not found
