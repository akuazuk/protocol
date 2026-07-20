"""Контроль качества клинических выдержек для карточки протокола.

Часть Summary Cards извлечена автоматически (`auto_extracted`) и содержит мусор:
одиночные слова («УЗИ», «контроль»), обрывки с номером пункта («4.1. малые критерии...
(далее - малый»), склейки списков с переносами. Этот модуль оставляет только
осмысленный, целостный клинический текст (полная фраза/предложение), иначе выдержку
отбрасываем и уходим в фолбэк на чистую прозу из чанков.

Детерминированно, без LLM.
"""
from __future__ import annotations

import re

from clinical_knowledge.meaningful_excerpt import meaningful_excerpt, normalize_text

_ENUM_PREFIX = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]\s*)+")
_BULLET_PREFIX = re.compile(r"^\s*[-–—•·*]+\s*")
_ALPHA_WORD = re.compile(r"[а-яёa-z]{3,}", re.I)
_LETTER = re.compile(r"[а-яёa-z]", re.I)

_MIN_CHARS = 24
_MIN_WORDS = 4
_MIN_LETTER_RATIO = 0.55

_STOP_TAILS = ("далее", "или", "и", "с", "в", "на", "по", "для", "при", "до")


def clean_clinical_text(text: str | None) -> str:
    """Нормализация: убрать переносы, ведущую нумерацию/маркеры, лишние пробелы."""
    t = normalize_text(text)
    if not t:
        return ""
    t = _ENUM_PREFIX.sub("", t)
    t = _BULLET_PREFIX.sub("", t)
    return t.strip()


def _trim_dangling(t: str) -> str:
    """Обрезать незакрытую скобку и висящий предлог/союз в конце."""
    if t.count("(") > t.count(")"):
        idx = t.rfind("(")
        if idx > 0:
            t = t[:idx].strip()
    t = t.rstrip(" ,;:-–—")
    words = t.split()
    while words and words[-1].lower().strip(".,;:") in _STOP_TAILS:
        words.pop()
    return " ".join(words).strip()


def is_meaningful_clinical_text(text: str | None) -> bool:
    """True, если текст читается как целостная клиническая фраза, а не обрывок."""
    t = clean_clinical_text(text)
    if len(t) < _MIN_CHARS:
        return False
    words = _ALPHA_WORD.findall(t)
    if len(words) < _MIN_WORDS:
        return False
    letters = len(_LETTER.findall(t))
    if letters / max(1, len(t)) < _MIN_LETTER_RATIO:
        return False
    return True


def meaningful_clinical_excerpt(text: str | None, *, limit: int = 240) -> str:
    """Вернуть чистую целостную выдержку в пределах limit или пустую строку.

    Пытается закончить на границе предложения; отбрасывает обрывки и мусор.
    """
    t = clean_clinical_text(text)
    if not t:
        return ""
    if len(t) > limit:
        t = meaningful_excerpt(t, limit=limit) or t[:limit]
        t = clean_clinical_text(t)
    t = _trim_dangling(t)
    if not is_meaningful_clinical_text(t):
        return ""
    return t


def best_meaningful_excerpt(candidates: list[str | None], *, limit: int = 240) -> str:
    """Первый осмысленный вариант из списка кандидатов (text, quote, ...)."""
    for cand in candidates:
        out = meaningful_clinical_excerpt(cand, limit=limit)
        if out:
            return out
    return ""
