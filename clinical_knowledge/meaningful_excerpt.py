"""Осмысленные выдержки текста без обрыва на полуслове."""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"[;:]+\s+")


def normalize_text(text: str | None) -> str:
    return _WS.sub(" ", (text or "").strip())


def split_sentences(text: str) -> list[str]:
    t = normalize_text(text)
    if not t:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(t) if p.strip()]
    return parts or [t]


def meaningful_excerpt(
    text: str | None,
    *,
    limit: int = 360,
    min_chars: int = 12,
) -> str:
    """Вернуть 1-3 целых предложения в пределах limit или пустую строку."""
    t = normalize_text(text)
    if len(t) < min_chars:
        return ""
    if len(t) <= limit:
        return t

    picked: list[str] = []
    used = 0
    for sent in split_sentences(t):
        if not sent:
            continue
        extra = len(sent) + (1 if picked else 0)
        if used + extra <= limit:
            picked.append(sent)
            used += extra
        else:
            break

    if picked:
        return " ".join(picked)

    first = split_sentences(t)[0]
    if len(first) <= limit:
        return first

    for chunk in _CLAUSE_SPLIT.split(first):
        chunk = chunk.strip()
        if chunk and len(chunk) <= limit:
            return chunk
    words = first.split()
    out: list[str] = []
    n = 0
    for w in words:
        add = len(w) + (1 if out else 0)
        if n + add > limit - 1:
            break
        out.append(w)
        n += add
    return " ".join(out) if out else ""


def excerpt_or_empty(text: str | None, *, limit: int = 360) -> str:
    """Выдержка только если текст содержательный."""
    raw = normalize_text(text)
    if not raw or raw.lower() in {"undefined", "нет", "не указано", "—", "-"}:
        return ""
    return meaningful_excerpt(raw, limit=limit)
