"""Локатор страницы PDF по дословной цитате из Summary Card.

Сопоставляет цитату пункта (verbatim quote) с текстом RAG-чанков протокола и
возвращает номер страницы (page_from). Нужен, чтобы дополнить выдержки карточки
ссылкой «стр. N», когда в самой карточке `page_start` пуст.

Стратегия: сначала точное вхождение устойчивых подстрок цитаты (начало/середина/
конец), затем - запасной вариант по доле совпавших токенов. Детерминированно, без LLM.
"""
from __future__ import annotations

import re
from typing import Any

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-zа-я0-9]{4,}")


def _norm(t: str | None) -> str:
    return _WS.sub(" ", (t or "").lower()).strip()


def _chunk_page(ch: dict[str, Any]) -> int | None:
    for k in ("page_from", "page", "page_start"):
        v = ch.get(k)
        try:
            p = int(v)
        except (TypeError, ValueError):
            continue
        if p > 0:
            return p
    return None


def _tokens(t: str | None) -> set[str]:
    return set(_TOKEN.findall(_norm(t)))


def locate_page_for_quote(
    quote: str | None,
    chunks: list[dict[str, Any]] | None,
    *,
    min_probe_len: int = 20,
    min_token_ratio: float = 0.6,
) -> int | None:
    """Вернуть номер страницы чанка, содержащего цитату (или наиболее близкого)."""
    q = _norm(quote)
    if len(q) < min_probe_len or not chunks:
        return None

    probes: list[str] = [q[:70]]
    if len(q) > 100:
        mid = len(q) // 2
        probes.append(q[mid - 35 : mid + 35])
    probes.append(q[-70:])
    probes = [p.strip() for p in probes if len(p.strip()) >= min_probe_len]

    norm_chunks: list[tuple[str, dict[str, Any]]] = []
    for ch in chunks:
        txt = _norm(ch.get("text"))
        if txt:
            norm_chunks.append((txt, ch))

    for txt, ch in norm_chunks:
        for probe in probes:
            if probe in txt:
                return _chunk_page(ch)

    qt = _tokens(quote)
    if len(qt) >= 4:
        best: dict[str, Any] | None = None
        best_ratio = 0.0
        for _txt, ch in norm_chunks:
            ct = _tokens(ch.get("text"))
            if not ct:
                continue
            ratio = len(qt & ct) / len(qt)
            if ratio > best_ratio:
                best_ratio = ratio
                best = ch
        if best is not None and best_ratio >= min_token_ratio:
            return _chunk_page(best)

    return None
