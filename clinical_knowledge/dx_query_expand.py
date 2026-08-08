"""Расширение текста диагноза для подбора КП (без поиска «по коду МКБ случая»).

Алиасы и нормализация формулировок врача → поисковый query.
Коды МКБ из текста случая вырезаются: они не должны вести матч карточек.
Кандидаты МКБ из справочника по русскому тексту - только lexical bridge
(RU title / членство кода на карточке КП), не gate по mis_diagnos.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

# МКБ-токены в свободном тексте (латиница/кириллица заглавная)
_ICD_TOKEN_RE = re.compile(
    r"\b[A-TV-ZА-Яа-я]\d{2}(?:[.,]\d{1,4})?\b",
    re.IGNORECASE,
)

# Длинные/специфичные раньше коротких
_DX_ALIAS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bпвус\b", re.IGNORECASE),
        "плосковальгусная установка стоп вальгусная деформация плоская стопа",
    ),
    (
        re.compile(r"плосковальгусн\w*", re.IGNORECASE),
        "плосковальгусная установка стоп вальгусная деформация плоская стопа pes planus",
    ),
    (
        re.compile(r"плоскостопи\w*", re.IGNORECASE),
        "плоская стопа pes planus деформация стоп",
    ),
    (
        re.compile(r"вальгусн\w*\s+деформац\w*", re.IGNORECASE),
        "вальгусная деформация плоская стопа установка стоп",
    ),
    (
        re.compile(r"вальгирован\w*", re.IGNORECASE),
        "вальгусная деформация",
    ),
    (
        re.compile(r"\bорви\b", re.IGNORECASE),
        "острая инфекция верхних дыхательных путей",
    ),
    (
        re.compile(r"\bгэрб\b|\bgerd\b", re.IGNORECASE),
        "гастроэзофагеальный рефлюкс",
    ),
    (
        re.compile(r"\bхобл\b", re.IGNORECASE),
        "хроническая обструктивная болезнь легких",
    ),
]

_STOP = frozenset(
    """
    диагноз код мкб болезнь заболевания пациент пациентка без при или
    не классифицированная других рубриках неуточненный неуточненная
    нарушения нарушен наруше опоры передвижения передвижен передвиже
    костей кость пяточных пяточн области других общие требования
    """.split()
)

# Порог lexicon для bridge (ниже - шум вроде «нарушения …»)
_BRIDGE_MIN_SCORE = 5.5


def strip_icd_tokens(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", _ICD_TOKEN_RE.sub(" ", text)).strip()


def matched_alias_phrases(text: str) -> list[str]:
    low = (text or "").lower()
    out: list[str] = []
    for pattern, phrase in _DX_ALIAS_PATTERNS:
        if pattern.search(low):
            out.append(phrase)
    return out


def expand_diagnosis_query(text: str) -> str:
    """Текст Dx для lexical match: без кодов МКБ + раскрытые алиасы."""
    raw = strip_icd_tokens(text or "")
    if not raw:
        return ""
    expansions: list[str] = [raw]
    expansions.extend(matched_alias_phrases(raw))
    return re.sub(r"\s+", " ", " ".join(expansions)).strip()


@lru_cache(maxsize=128)
def _bridge_icd_candidates_cached(
    raw_key: str,
    min_score: float,
    limit: int,
) -> tuple[tuple[str, str, float], ...]:
    """Cached text→ICD; key = stripped/expanded diagnosis text."""
    raw = raw_key
    if not raw:
        return ()
    queries: list[str] = []
    expanded = expand_diagnosis_query(raw)
    if expanded:
        queries.append(expanded)
    for phrase in matched_alias_phrases(raw):
        if phrase not in queries:
            queries.append(phrase)
        # короткие нозологические nuggets из alias (иначе lexicon «размазывается»)
        for nugget in (
            "вальгусная деформация",
            "плоская стопа",
            "pes planus",
            "плосковальгусная установка стоп",
            "острая инфекция верхних дыхательных путей",
            "гастроэзофагеальный рефлюкс",
            "хроническая обструктивная болезнь легких",
        ):
            if nugget in phrase.lower() and nugget not in queries:
                queries.append(nugget)
    # focused alias-free core
    core = raw[:180]
    if core and core not in queries:
        queries.append(core)

    try:
        import icd_mkb
    except Exception:  # noqa: BLE001
        return ()

    best: dict[str, tuple[str, str, float]] = {}
    for query in queries:
        try:
            rows = icd_mkb.suggest_icd_from_russian(query, max_results=12)
        except Exception:  # noqa: BLE001
            continue
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            code = str(row.get("code") or "").strip().upper()
            score = float(row.get("score") or 0.0)
            if not code or score < min_score:
                continue
            title = str(row.get("title_ru") or "").strip()
            prev = best.get(code)
            if prev is None or score > prev[2]:
                best[code] = (code, title, score)
    ranked = sorted(best.values(), key=lambda item: -item[2])
    if not ranked:
        return ()
    top = ranked[0][2]
    # держим только близких к лидеру - отсекает C39/R04/J66 при ОРВИ
    floor = max(min_score, top * 0.55)
    tight = [row for row in ranked if row[2] >= floor]
    return tuple((tight or ranked)[:limit])


def _bridge_code_plausible(code: str, title: str, expanded_low: str) -> bool:
    """Грубый фильтр глав МКБ, нерелевантных клиническому Dx-тексту."""
    ch = (code or "")[:1]
    title_l = (title or "").lower()
    if ch in {"C", "D", "V", "W", "X", "Y"}:
        return False
    if ch == "R" and any(x in expanded_low for x in ("инфекц", "орви", "грипп")):
        return False
    if ("моче" in title_l or "беременност" in title_l) and any(
        x in expanded_low for x in ("дыхат", "орви", "стоп", "вальгус", "плоско")
    ):
        return False
    if "инородн" in title_l and "инородн" not in expanded_low:
        return False
    if "злокачествен" in title_l and "злокачеств" not in expanded_low:
        return False
    return True


def bridge_icd_candidates(
    text: str,
    *,
    min_score: float = _BRIDGE_MIN_SCORE,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Кандидаты МКБ из русского текста диагноза (справочник), не из поля кода случая."""
    raw = strip_icd_tokens(text or "")
    if not raw:
        return []
    rows = _bridge_icd_candidates_cached(raw.lower(), float(min_score), int(limit))
    expanded_low = expand_diagnosis_query(raw).lower()
    cands = [
        {"code": c, "title_ru": t, "score": s}
        for c, t, s in rows
        if _bridge_code_plausible(c, t, expanded_low)
    ]
    return refine_bridge_candidates(cands, raw, limit=min(4, limit))


def enrich_query_with_bridge_titles(text: str) -> str:
    """Query + RU titles сильных text→ICD кандидатов (один раз на случай)."""
    base = expand_diagnosis_query(text)
    if not base:
        return ""
    titles = [
        str(item.get("title_ru") or "").strip()
        for item in bridge_icd_candidates(base)
        if item.get("title_ru")
    ]
    if not titles:
        return base
    cleaned: list[str] = []
    lead = re.compile(r"^\s*[A-Za-z]\d{2}(?:\.\d{1,4})?\s*[-:–—]\s*")
    for title in titles[:8]:
        cleaned.append(lead.sub("", title).strip())
    return re.sub(r"\s+", " ", (base + " " + " ".join(cleaned)).strip())


@lru_cache(maxsize=256)
def _diagnosis_tokens_cached(expanded_key: str, min_len: int, limit: int) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"[а-яёa-z0-9]{%d,}" % min_len, expanded_key):
        if word in _STOP:
            continue
        candidates = [word]
        if len(word) >= 7:
            candidates.append(word[:-2])
        if len(word) >= 9:
            candidates.append(word[:-3])
        for token in candidates:
            if len(token) < min_len or token in _STOP or token in seen:
                continue
            seen.add(token)
            found.append(token)
            if len(found) >= limit:
                return tuple(found)
    return tuple(found)


def diagnosis_tokens(text: str, *, min_len: int = 3, limit: int = 24) -> list[str]:
    """Токены + лёгкие стемы для overlap с карточками КП.

    Не вызывает lexicon ICD (дорого) - ожидается уже expand/enrich снаружи,
    либо достаточно alias-expand.
    """
    expanded = expand_diagnosis_query(text)
    if not expanded:
        return []
    return list(_diagnosis_tokens_cached(expanded.lower(), int(min_len), int(limit)))


_WEAK_TOKEN_PREFIXES = (
    "деформа",
    "заболева",
    "нарушен",
    "неуточн",
    "други",
    "общи",
    "лечени",
    "диагност",
    "пациент",
    "врожден",
    "приобрет",
    "дыхател",
    "путей",
    "верхни",
    "нижни",
    "инфекц",
    "острая",
    "остры",
    "болезн",
)


def token_weight(token: str) -> float:
    """Вес токена для overlap: длинные/нозологические важнее общих."""
    t = token or ""
    if any(t.startswith(p) for p in _WEAK_TOKEN_PREFIXES):
        return 0.3
    n = len(t)
    if n >= 10:
        return 2.0
    if n >= 6:
        return 1.25
    if n >= 4:
        return 1.0
    return 0.5


def refine_bridge_candidates(
    cands: list[dict[str, Any]],
    diag_text: str,
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Оставить text→ICD, чьи titles делят с Dx нозологические маркеры."""
    if not cands:
        return []
    expanded = expand_diagnosis_query(diag_text).lower()
    dx_tokens = {
        t
        for t in diagnosis_tokens(diag_text, min_len=4, limit=40)
        if token_weight(t) >= 1.0
    }
    markers = (
        "вальгус",
        "плоско",
        "плоск",
        "стоп",
        "planus",
        "пвус",
        "рефлюкс",
        "обструктивн",
        "верхних дыхательных путей",
        "орви",
    )
    strong: list[dict[str, Any]] = []
    lead = re.compile(r"^\s*[A-Za-z]\d{2}(?:\.\d{1,4})?\s*[-:\u2013\u2014]\s*")
    for item in cands:
        title = lead.sub("", str(item.get("title_ru") or "")).lower()
        if not title:
            continue
        title_tokens = {
            tok for tok in re.findall(r"[а-яёa-z0-9]{4,}", title) if token_weight(tok) >= 1.0
        }
        share = dx_tokens & title_tokens
        marker_hit = any(m in title for m in markers) and any(m in expanded for m in markers)
        if share or marker_hit:
            strong.append(item)
    chosen = strong if strong else list(cands)[:2]
    return chosen[:limit]
