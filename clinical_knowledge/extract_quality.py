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

_BRACKET_PREFIX = re.compile(r"^\s*\[[^\]]{1,40}\]\s*")
_ENUM_PREFIX = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]\s*)+")
_BULLET_PREFIX = re.compile(r"^\s*[-–—•·*]+\s*")
_ALPHA_WORD = re.compile(r"[а-яёa-z]{3,}", re.I)
_LETTER = re.compile(r"[а-яёa-z]", re.I)

# Строка-глоссарий: короткий токен-аббревиатура + тире + расшифровка.
# «АСИТ - аллергенспецифическая иммунотерапия», «MART-терапия - режим...», «SpO2 - ...».
_GLOSSARY_LINE = re.compile(r"^\S{1,16}\s*[–—-]\s+\S")
# Высокоточные записи словаря сокращений (даже длинные): латинская аббревиатура в
# начале или «ААА тест/терапия -». Не задевает содержательные «БА - гетерогенное...».
_GLOSSARY_ABBR = re.compile(
    r"^(?:[A-Za-z][A-Za-zА-Яа-яЁё0-9/-]{0,20}(?:\s(?:тест|терапия))?|[А-ЯЁ]{2,6}\s(?:тест|терапия))\s*[- - -]\s",
)

_MIN_CHARS = 24
_MIN_WORDS = 4
_MIN_LETTER_RATIO = 0.55

_STOP_TAILS = ("далее", "или", "и", "с", "в", "на", "по", "для", "при", "до")

# Юридический/оглавительный шум, не несущий клинического смысла.
_BOILERPLATE = (
    "термины и их определения",
    "о здравоохранении",
    "общие положения",
    "область применения",
    "перечень сокращен",
    "настоящий клинический протокол",
    "настоящего клинического протокола",
    "нормативные ссылки",
    "список литератур",
    "шифр по международной",
    "шифр по мкб",
    "международной статистической классификации болезн",
    "используются следующие сокращения",
    # Правовая обвязка постановлений/приказов Минздрава РБ (шапки, подписи, футеры).
    "национальный правовой интернет-портал",
    "признать утратившим силу",
    "признать утратившими силу",
    "областной исполнительный комитет",
    "городской исполнительный комитет",
    "минский городской исполнительный",
    "совет министров республики беларусь",
    "постановление министерства здравоохранения",
    "постановления министерства здравоохранения",
    "об утверждении клинического протокола",
    "об утверждении некоторых клинических протоколов",
    "утвердить клинический протокол",
    "утвердить прилагаем",
    "утвердить следующие",
    "настоящее постановление вступает в силу",
    "вступает в силу после его официального опубликования",
    "официального опубликования",
    "зарегистрировано в национальном реестре",
)

# Подпись министра «Министр Д.Л.Пиневич» и одиночный маркер главы «ГЛАВА 2».
_MINISTER_SIG_RE = re.compile(r"министр\s+[а-яё]\.\s?[а-яё]\.\s?[а-яё]", re.I)
_CHAPTER_ONLY_RE = re.compile(r"^\s*глава\s+\d+\s*$", re.I)


def is_legal_admin_text(text: str | None) -> bool:
    """True для юридической/административной обвязки (не клинический смысл)."""
    t = normalize_text(text)
    if not t:
        return False
    low = t.lower()
    if any(bp in low for bp in _BOILERPLATE):
        return True
    if _MINISTER_SIG_RE.search(low):
        return True
    if _CHAPTER_ONLY_RE.match(t):
        return True
    return False


def clean_clinical_text(text: str | None) -> str:
    """Нормализация: убрать переносы, теговые/скобочные и нумерационные префиксы, пробелы."""
    t = normalize_text(text)
    if not t:
        return ""
    prev = None
    while prev != t:
        prev = t
        t = _BRACKET_PREFIX.sub("", t)
        t = _ENUM_PREFIX.sub("", t)
        t = _BULLET_PREFIX.sub("", t)
    return t.strip()


def _trim_dangling(t: str) -> str:
    """Обрезать незакрытую скобку и висящий предлог/союз в конце."""
    if t.count("(") > t.count(")"):
        idx = t.rfind("(")
        if idx > 0:
            t = t[:idx].strip()
    t = t.rstrip(" ,;:-- - ")
    words = t.split()
    while words and words[-1].lower().strip(".,;:") in _STOP_TAILS:
        words.pop()
    return " ".join(words).strip()


def starts_like_sentence(text: str | None) -> bool:
    """True, если строка начинается как предложение (заглавная/цифра/кавычка).

    Отсекает обрывки, начинающиеся с середины слова или предлога/союза:
    «ьзование…», «население)…», «или средства…», «по систематической…».
    """
    t = clean_clinical_text(text)
    if not t:
        return False
    ch = t[0]
    if ch.isdigit() or ch in "«\"“":
        return True
    if ch.isalpha():
        return ch.isupper()
    return False


def is_meaningful_clinical_text(text: str | None, *, require_sentence_start: bool = False) -> bool:
    """True, если текст читается как целостная клиническая фраза, а не обрывок."""
    t = clean_clinical_text(text)
    if len(t) < _MIN_CHARS:
        return False
    if require_sentence_start and not starts_like_sentence(t):
        return False
    words = _ALPHA_WORD.findall(t)
    if len(words) < _MIN_WORDS:
        return False
    letters = len(_LETTER.findall(t))
    if letters / max(1, len(t)) < _MIN_LETTER_RATIO:
        return False
    if is_legal_admin_text(t):
        return False
    if _GLOSSARY_LINE.match(t) and len(words) < 8:
        return False
    if _GLOSSARY_ABBR.match(t):
        return False
    return True


def _dedup_key(text: str) -> tuple[str, frozenset[str]]:
    """Нормализованная форма + множество значимых токенов для сравнения дублей."""
    low = normalize_text(text).lower().replace("ё", "е")
    tokens = frozenset(w for w in _ALPHA_WORD.findall(low) if len(w) >= 4)
    norm = re.sub(r"[^а-яa-z0-9 ]+", " ", low)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm, tokens


def _is_near_duplicate(norm: str, tokens: frozenset[str], kept: list[tuple[str, frozenset[str]]]) -> bool:
    for kn, kt in kept:
        if not norm or not kn:
            continue
        if norm == kn or norm in kn or kn in norm:
            return True
        if tokens and kt:
            inter = len(tokens & kt)
            union = len(tokens | kt)
            if union and inter / union >= 0.72:
                return True
    return False


class Deduper:
    """Стейтфул-дедуп: накапливает принятые фразы, отбивает точные и near-дубли.

    Нужен когда рядом с текстом надо сохранить метаданные (цитату, страницу) - тогда
    `dedupe_meaningful` (только тексты) не подходит.
    """

    def __init__(self) -> None:
        self._kept: list[tuple[str, frozenset[str]]] = []

    def accept(self, text: str) -> bool:
        """True и запоминает, если text - не дубль ранее принятых; иначе False."""
        norm, tokens = _dedup_key(text)
        if not norm:
            return False
        if _is_near_duplicate(norm, tokens, self._kept):
            return False
        self._kept.append((norm, tokens))
        return True


def new_deduper() -> Deduper:
    return Deduper()


def dedupe_meaningful(
    candidates: list[str | None],
    *,
    limit: int = 240,
    max_items: int = 12,
) -> list[str]:
    """Осмысленные выдержки без дублей и near-дублей, с сохранением порядка.

    Для навигатора: убирает повторяющиеся лиды («Диагностика и лечение.» ×3) и
    почти совпадающие фразы (Jaccard по токенам), оставляя более полную формулировку.
    """
    out: list[str] = []
    kept: list[tuple[str, frozenset[str]]] = []
    for cand in candidates:
        if len(out) >= max_items:
            break
        text = meaningful_clinical_excerpt(cand, limit=limit)
        if not text:
            continue
        norm, tokens = _dedup_key(text)
        if _is_near_duplicate(norm, tokens, kept):
            continue
        out.append(text)
        kept.append((norm, tokens))
    return out


def meaningful_clinical_excerpt(
    text: str | None,
    *,
    limit: int = 240,
    require_sentence_start: bool = False,
) -> str:
    """Вернуть чистую целостную выдержку в пределах limit или пустую строку.

    Пытается закончить на границе предложения; отбрасывает обрывки и мусор.
    С `require_sentence_start` также отбрасывает фразы, начатые не с начала
    предложения (обрывки с середины слова/после предлога).
    """
    t = clean_clinical_text(text)
    if not t:
        return ""
    if len(t) > limit:
        t = meaningful_excerpt(t, limit=limit) or t[:limit]
        t = clean_clinical_text(t)
    t = _trim_dangling(t)
    if not is_meaningful_clinical_text(t, require_sentence_start=require_sentence_start):
        return ""
    return t


def best_meaningful_excerpt(
    candidates: list[str | None],
    *,
    limit: int = 240,
    require_sentence_start: bool = False,
) -> str:
    """Первый осмысленный вариант из списка кандидатов (text, quote, ...)."""
    for cand in candidates:
        out = meaningful_clinical_excerpt(
            cand, limit=limit, require_sentence_start=require_sentence_start
        )
        if out:
            return out
    return ""
