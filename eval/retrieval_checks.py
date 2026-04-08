"""
Общие проверки качества retrieve() — без импорта rag_server (удобно для pytest).
"""
from __future__ import annotations

from typing import Any

REQUIRED_RESULT_KEYS = frozenset({"path", "excerpt", "kind", "score"})


def combined_chunk_text(retrieved: list[dict]) -> str:
    parts: list[str] = []
    for r in retrieved:
        parts.append(str(r.get("excerpt") or ""))
        parts.append(str(r.get("title") or ""))
        parts.append(str(r.get("path") or ""))
    return "\n".join(parts).lower()


def check_must_substrings(
    retrieved: list[dict], must: list[str]
) -> tuple[bool, list[str]]:
    if not must:
        return True, []
    hay = combined_chunk_text(retrieved)
    missing: list[str] = []
    for s in must:
        if not s:
            continue
        if str(s).lower() not in hay:
            missing.append(s)
    return len(missing) == 0, missing


def check_expected_paths(
    retrieved: list[dict], fragments: list[str]
) -> tuple[bool, str | None]:
    if not fragments:
        return True, None
    paths = [str(r.get("path") or "").lower() for r in retrieved]
    if not paths:
        return False, "нет путей (пустой отбор)"
    for frag in fragments:
        f = str(frag).lower().strip()
        if not f:
            continue
        for p in paths:
            if f in p:
                return True, None
    return False, "ни один топ-путь не содержит ожидаемых фрагментов"


def check_forbidden_substrings(
    retrieved: list[dict], forbidden: list[str]
) -> tuple[bool, list[str]]:
    """True, если ни одна из запрещённых подстрок не встречается в объединённом тексте."""
    if not forbidden:
        return True, []
    hay = combined_chunk_text(retrieved)
    bad: list[str] = []
    for s in forbidden:
        if not s:
            continue
        if str(s).lower() in hay:
            bad.append(s)
    return len(bad) == 0, bad


def validate_retrieval_schema(rows: list[dict]) -> list[str]:
    """Сообщения об ошибках схемы; пустой список = ок."""
    errs: list[str] = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            errs.append(f"[{i}] не dict")
            continue
        missing = REQUIRED_RESULT_KEYS - set(r.keys())
        if missing:
            errs.append(f"[{i}] нет ключей: {sorted(missing)}")
        sc = r.get("score")
        if sc is not None and not isinstance(sc, (int, float)):
            errs.append(f"[{i}] score не число")
        elif isinstance(sc, (int, float)) and sc < 0:
            errs.append(f"[{i}] score < 0")
    return errs


def score_metrics(rows: list[dict]) -> dict[str, Any]:
    scores = [float(r.get("score") or 0.0) for r in rows if isinstance(r, dict)]
    if not scores:
        return {
            "top_score": None,
            "second_score": None,
            "spread": None,
            "n": 0,
        }
    s_sorted = sorted(scores, reverse=True)
    top = s_sorted[0]
    second = s_sorted[1] if len(s_sorted) > 1 else None
    spread = (top - second) if second is not None else None
    return {
        "top_score": round(top, 6),
        "second_score": round(second, 6) if second is not None else None,
        "spread": round(spread, 6) if spread is not None else None,
        "n": len(scores),
    }


def path_rank(retrieved: list[dict], path_fragment: str) -> int | None:
    """1-based позиция первого чанка, чей path содержит fragment; None если нет."""
    f = str(path_fragment).lower()
    if not f:
        return None
    for i, r in enumerate(retrieved, 1):
        p = str(r.get("path") or "").lower()
        if f in p:
            return i
    return None
