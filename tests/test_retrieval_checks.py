"""Юнит-тесты eval/retrieval_checks.py (без rag_server)."""
from __future__ import annotations

from eval.retrieval_checks import (
    check_expected_paths,
    check_forbidden_substrings,
    check_must_substrings,
    path_rank,
    score_metrics,
    validate_retrieval_schema,
)


def test_validate_retrieval_schema_ok() -> None:
    rows = [
        {"path": "a.pdf", "excerpt": "x", "kind": "body", "score": 0.5},
    ]
    assert validate_retrieval_schema(rows) == []


def test_validate_retrieval_schema_missing_key() -> None:
    rows = [{"path": "a.pdf", "kind": "body"}]
    errs = validate_retrieval_schema(rows)
    assert errs and "excerpt" in errs[0]


def test_check_must_substrings() -> None:
    r = [{"excerpt": "Острый бронхит J20", "title": "", "path": "p.pdf"}]
    ok, miss = check_must_substrings(r, ["бронхит"])
    assert ok and miss == []
    ok2, miss2 = check_must_substrings(r, ["гипертония"])
    assert not ok2 and miss2 == ["гипертония"]


def test_check_expected_paths() -> None:
    r = [{"path": "minzdrav/x/y_smoke.pdf", "excerpt": ""}]
    ok, _ = check_expected_paths(r, ["_smoke"])
    assert ok
    ok2, detail = check_expected_paths(r, ["nope"])
    assert not ok2 and detail


def test_forbidden() -> None:
    r = [{"excerpt": "текст про диабет", "path": "p"}]
    ok, bad = check_forbidden_substrings(r, ["диабет"])
    assert not ok and bad == ["диабет"]


def test_score_metrics() -> None:
    m = score_metrics(
        [
            {"score": 0.5},
            {"score": 0.3},
        ]
    )
    assert m["top_score"] == 0.5
    assert m["spread"] is not None and m["spread"] > 0


def test_path_rank() -> None:
    r = [
        {"path": "a/first.pdf"},
        {"path": "b/second.pdf"},
    ]
    assert path_rank(r, "second") == 2
    assert path_rank(r, "none") is None
