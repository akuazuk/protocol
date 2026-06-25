"""Семантический fallback для правил required_exam / keyword_presence (после substring)."""
from __future__ import annotations

import os
import re
from typing import Any

# Распространённые синонимы/сокращения в КЗ (без вызова API).
_ALIAS_MAP: dict[str, list[str]] = {
    "узи": ["ультразвук", "сонограф", "эхограф"],
    "эхокг": ["эхо-кг", "эхокардиограф", "узи сердца"],
    "кт": ["компьютерн", "томограф"],
    "мрт": ["магнитно-резонанс", "мр-"],
    "фгдс": ["эзофагогастродуоденоскоп", "гастроскоп", "эгдс"],
    "эгдс": ["эзофагогастродуоденоскоп", "гастроскоп", "фгдс"],
    "ривароксабан": ["ксарелто", "ксабан"],
    "апиксабан": ["эликвис"],
    "колоноскоп": ["фкс", "ирригоскоп"],
    "экг": ["электрокардиограф", "регистрац"],
    "холтер": ["суточн", "мониторирован", "экг"],
    "дуплекс": ["допплер", "сканирован", "вен нижн"],
    "анализ крови": ["оак", "общий анализ крови", "гемоглобин", "лейкоц"],
    "коагулограм": ["мно", "протромбин", "фибриноген"],
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _token_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[а-яёa-z0-9]{3,}", _norm(s)) if len(t) >= 3}


def _aliases(term: str) -> list[str]:
    low = _norm(term)
    out = [low] if low else []
    for key, vals in _ALIAS_MAP.items():
        if key in low or low in key:
            out.extend([key] + vals)
        for v in vals:
            if v in low or low in v:
                out.extend([key] + vals)
    extra = list(term) if isinstance(term, str) else []
    for x in extra:
        sx = _norm(str(x))
        if sx and sx not in out:
            out.append(sx)
    seen: set[str] = set()
    uniq: list[str] = []
    for a in out:
        if a and a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


def fuzzy_term_in_text(text: str, term: str) -> tuple[bool, float, str | None]:
    """Проверка по алиасам и пересечению токенов (без API)."""
    low = _norm(text)
    if not low or not term:
        return False, 0.0, None
    for alias in _aliases(term):
        if len(alias) >= 4 and alias in low:
            return True, 0.92, alias
    term_tokens = _token_set(term)
    if len(term_tokens) < 2:
        return False, 0.0, None
    text_tokens = _token_set(text)
    if not text_tokens:
        return False, 0.0, None
    overlap = term_tokens & text_tokens
    ratio = len(overlap) / max(len(term_tokens), 1)
    if ratio >= 0.6:
        return True, round(0.75 + 0.2 * ratio, 3), ", ".join(sorted(overlap)[:4])
    return False, ratio, None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def embedding_term_match(
    text: str,
    term: str,
    *,
    embed_fn: Any = None,
) -> tuple[bool, float, str | None]:
    """Опциональный embed-match: один вызов на пару (KZ excerpt, term)."""
    if not _env_bool("RULE_SEMANTIC_EMBED", False):
        return False, 0.0, None
    excerpt = (text or "")[:4000].strip()
    if len(excerpt) < 40 or not term:
        return False, 0.0, None
    if embed_fn is None:
        return False, 0.0, None
    try:
        q_vec, t_vec = embed_fn(excerpt, term)
    except Exception:
        return False, 0.0, None
    cos = _cosine(q_vec, t_vec)
    threshold = float(os.environ.get("RULE_SEMANTIC_EMBED_THRESHOLD", "0.82"))
    if cos >= threshold:
        return True, round(cos, 3), f"embedding:{term[:60]}"
    return False, cos, None


def semantic_presence_check(
    text: str,
    term: str,
    *,
    rule: dict[str, Any] | None = None,
    embed_fn: Any = None,
) -> dict[str, Any]:
    """Второй проход после substring: aliases → optional embedding."""
    if not _env_bool("RULE_SEMANTIC_FALLBACK", True):
        return {"matched": False, "method": "disabled"}
    low = _norm(text)
    needle = _norm(term)
    if needle and needle in low:
        return {"matched": True, "method": "substring", "confidence": 1.0}

    aliases = list(rule.get("semantic_aliases") or []) if rule else []
    for alias in _aliases(term) + [_norm(a) for a in aliases if a]:
        if len(alias) >= 4 and alias in low:
            return {
                "matched": True,
                "method": "alias",
                "confidence": 0.9,
                "matched_alias": alias,
            }

    ok, conf, hint = fuzzy_term_in_text(text, term)
    if ok:
        return {
            "matched": True,
            "method": "fuzzy_tokens",
            "confidence": conf,
            "matched_alias": hint,
        }

    ok, conf, hint = embedding_term_match(text, term, embed_fn=embed_fn)
    if ok:
        return {
            "matched": True,
            "method": "embedding",
            "confidence": conf,
            "matched_alias": hint,
        }

    return {"matched": False, "method": "none", "confidence": conf}
