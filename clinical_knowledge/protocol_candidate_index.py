"""Префильтр карточек КП: токены запроса и корни МКБ, без полного score 6600 карт."""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

from clinical_knowledge.dx_query_expand import diagnosis_tokens, expand_diagnosis_query
from clinical_knowledge.kp_validity import looks_omnibus

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]{4,}")


def _icd_root(code: str) -> str:
    text = (code or "").upper().strip()
    return text[:3] if len(text) >= 3 else text


def _card_index_blob(card: dict[str, Any]) -> str:
    # Только паспорт карты. Тело PDF даёт чужие КП (ПЦД, ГСК, омнибус).
    parts = [
        str(card.get("title") or ""),
        str(card.get("condition_label") or ""),
        str(card.get("source_path") or "").replace("_", " ").replace("-", " ").replace("/", " "),
    ]
    return " ".join(parts).lower()


@lru_cache(maxsize=1)
def _built_index() -> tuple[dict[str, frozenset[int]], dict[str, frozenset[int]]]:
    from clinical_knowledge.loader import load_protocol_cards_registry

    token_map: dict[str, set[int]] = defaultdict(set)
    icd_map: dict[str, set[int]] = defaultdict(set)
    cards = load_protocol_cards_registry()
    for idx, card in enumerate(cards):
        blob = _card_index_blob(card)
        for token in _TOKEN_RE.findall(blob):
            token_map[token].add(idx)
        primary = [str(x).strip().upper() for x in (card.get("icd10_primary") or []) if x]
        extra = []
        if not looks_omnibus(card):
            extra = [str(x).strip().upper() for x in (card.get("icd10_all") or []) if x]
        codes = list(dict.fromkeys(primary + extra))
        for code in codes:
            icd_map[code].add(idx)
            root = _icd_root(code)
            if root:
                icd_map[root].add(idx)
    return (
        {key: frozenset(val) for key, val in token_map.items()},
        {key: frozenset(val) for key, val in icd_map.items()},
    )


def select_candidate_cards(
    cards: list[dict[str, Any]] | None = None,
    *,
    diag_text: str,
    icd_list: list[str],
    specialty_slug: str | None = None,
) -> list[dict[str, Any]]:
    """Подмножество карточек, по которым есть lexical или МКБ сигнал.

    Индекс строится по полному реестру; переданный ``cards`` не используем
    как массив индексов (после фильтра по специальности номера съезжают).
    """
    from clinical_knowledge.loader import load_protocol_cards_registry

    registry = load_protocol_cards_registry()
    token_map, icd_map = _built_index()
    hits: set[int] = set()
    expanded = expand_diagnosis_query(diag_text) if diag_text else ""
    for token in diagnosis_tokens(expanded or diag_text, min_len=4, limit=28):
        found = token_map.get(token)
        if found:
            hits |= set(found)
    for raw in icd_list:
        code = str(raw or "").strip().upper()
        if not code:
            continue
        found = icd_map.get(code)
        if found:
            hits |= set(found)
        root = _icd_root(code)
        if root and root != code:
            found = icd_map.get(root)
            if found:
                hits |= set(found)
    if not hits:
        return []
    out = [registry[idx] for idx in sorted(hits) if 0 <= idx < len(registry)]
    if specialty_slug:
        return [card for card in out if card.get("specialty_slug") == specialty_slug]
    _ = cards
    return out


def clear_candidate_index() -> None:
    _built_index.cache_clear()
