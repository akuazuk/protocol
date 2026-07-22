"""Каталог синонимов обследований/препаратов для семантического матча (Э2).

Загружает data/catalog/exam_drug_synonyms.json (собран scripts/build_exam_drug_synonyms.py)
и строит индексы variant↔canonical + инвертированный токен-индекс. Используется
semantic_rule_fallback, чтобы термин протокола («Общий анализ крови развернутый»)
матчился с сокращением в КЗ («ОАК») и наоборот.

Без внешних зависимостей (чистый stdlib) - грузится в любом окружении.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CATALOG_PATH = _ROOT / "data" / "catalog" / "exam_drug_synonyms.json"

# минимальная длина токена/алиаса, чтобы не давать ложных подстрок ("кт" - исключение)
_MIN_ALIAS_LEN = 4
_SHORT_WHITELIST = {"кт", "ркт", "мрт", "экг", "оак", "оам", "бак", "узи", "соэ", "ттг",
                    "пса", "фгдс", "эгдс", "ээг", "энмг", "фвд", "ипп", "нпвс", "фкс",
                    "смад", "флг", "уздг", "гкс", "иапф", "бра", "баб", "бкк", "абт",
                    "рг", "мг", "мно", "ачтв", "пти", "эхокг"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().replace("ё", "е")).strip()


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[а-яa-z0-9]{3,}", _norm(s))}


class _CatalogIndex:
    __slots__ = ("variant_to_canon", "canon_to_variants", "token_to_canons", "empty")

    def __init__(self) -> None:
        self.variant_to_canon: dict[str, str] = {}
        self.canon_to_variants: dict[str, list[str]] = {}
        self.token_to_canons: dict[str, set[str]] = {}
        self.empty = True

    def add_group(self, mapping: dict[str, list[str]]) -> None:
        for canon_raw, variants in mapping.items():
            canon = _norm(canon_raw)
            if not canon:
                continue
            self.empty = False
            all_forms = {canon} | {_norm(v) for v in (variants or []) if _norm(v)}
            self.canon_to_variants.setdefault(canon, [])
            for form in all_forms:
                if form and form not in self.variant_to_canon:
                    self.variant_to_canon[form] = canon
                if form != canon and form not in self.canon_to_variants[canon]:
                    self.canon_to_variants[canon].append(form)
            for tok in _tokens(canon):
                self.token_to_canons.setdefault(tok, set()).add(canon)


@lru_cache(maxsize=1)
def _index() -> _CatalogIndex:
    idx = _CatalogIndex()
    try:
        data = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return idx
    if isinstance(data.get("exams"), dict):
        idx.add_group(data["exams"])
    if isinstance(data.get("drug_groups"), dict):
        idx.add_group(data["drug_groups"])
    return idx


def catalog_available() -> bool:
    return not _index().empty


def _resolve_canonical(term_norm: str, idx: _CatalogIndex) -> str | None:
    """К какому канону относится term (точное совпадение / подстрока / общий токен)."""
    if not term_norm:
        return None
    hit = idx.variant_to_canon.get(term_norm)
    if hit:
        return hit
    # кандидаты по общим токенам (ограничивает перебор)
    cand: set[str] = set()
    for tok in _tokens(term_norm):
        cand |= idx.token_to_canons.get(tok, set())
    best: str | None = None
    best_len = 0
    for canon in cand:
        # канон целиком содержится в термине (термин = «общий анализ крови развернутый»)
        if canon in term_norm and len(canon) > best_len:
            best, best_len = canon, len(canon)
    if best:
        return best
    # обратная подстрока: короткий термин содержится в варианте (длинные имена протоколов
    # вариантами быть не могут - пропускаем дорогой скан)
    if _MIN_ALIAS_LEN <= len(term_norm) <= 24:
        for form, canon in idx.variant_to_canon.items():
            if term_norm in form:
                return canon
    return None


@lru_cache(maxsize=4096)
def expand_term(term: str) -> tuple[str, ...]:
    """Все формы (канон + варианты) для термина; пусто, если нет в каталоге.

    Пример: expand_term("Общий анализ крови развернутый")
      -> ("общий анализ крови", "оак", "клинический анализ крови", ...)
    """
    idx = _index()
    if idx.empty:
        return ()
    tn = _norm(term)
    canon = _resolve_canonical(tn, idx)
    if not canon:
        return ()
    forms = {canon, *idx.canon_to_variants.get(canon, [])}
    out = [f for f in forms if len(f) >= _MIN_ALIAS_LEN or f in _SHORT_WHITELIST]
    return tuple(sorted(out, key=len, reverse=True))


def clear_cache() -> None:
    _index.cache_clear()
    expand_term.cache_clear()
