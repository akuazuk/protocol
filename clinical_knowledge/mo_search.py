"""Умный поиск по случаям МО: запрос -> план поиска (МКБ, синонимы, стеммы, опечатки).

Конвейер `expand_query(q)`:

1. нормализация (нижний регистр, ё->е, пунктуация -> пробел);
2. идентификаторы (`visit_id` / `patient_id`) остаются в `mo_backend._identity_lookup`;
3. коды МКБ `I10`, `I1`, `I10-I15`, `I10.` -> набор префиксов;
4. алиасы и синонимы из `data/icd_reference/dx_aliases_ru.json` -> фразы;
5. термин -> коды по русским названиям МКБ-10-СУ (стеммы), свёрнутые до префиксов (<= 40);
6. опечатки: триграммная близость токена к словарю слов названий МКБ (>= 0.6);
7. SQL / Python-предикат с разбором «по чему нашли» (`search_plan`) и чипами,
   которые можно выключить (`search_off=icd,synonyms,fuzzy,terms`).

Эмбеддинги «похожих» (п. 8 плана) сюда не входят: словарь для них строится ночью на GCE.
Модуль не ходит в БД: словарь `dim_diagnosis` при желании передаётся снаружи.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .mo_icd_aliases import _apply_word_expansions

ROOT = Path(__file__).resolve().parents[1]
ALIASES_PATH = ROOT / "data" / "icd_reference" / "dx_aliases_ru.json"
ICD_RU_PATH = ROOT / "data" / "icd_reference" / "icd10_ru_mkb10su.json"

MAX_CODE_PREFIXES = 40
MAX_PHRASES = 12
MAX_FUZZY = 3
FUZZY_MIN_SCORE = 0.6
FUZZY_MIN_TOKEN = 5
STEM_MIN = 4

CHIP_ICD = "icd"
CHIP_PHRASE = "phrase"
CHIP_TERMS = "terms"
CHIP_SYNONYMS = "synonyms"
CHIP_FUZZY = "fuzzy"
CHIP_DOCTOR = "doctor"
ALL_CHIPS = (CHIP_ICD, CHIP_PHRASE, CHIP_TERMS, CHIP_SYNONYMS, CHIP_FUZZY, CHIP_DOCTOR)

_ICD_CODE_RE = re.compile(r"^[A-Z]\d(?:\d(?:\.\d{0,2})?)?\.?$")
_ICD_RANGE_RE = re.compile(r"^([A-Z])(\d{2})\s*-\s*(?:([A-Z]))?(\d{2})$")
_TOKEN_RE = re.compile(r"[а-яa-z0-9]+")
# Кириллические двойники латиницы в кодах МКБ: «І10», «Е11», «К29» с русской раскладки.
_CYR_TO_LATIN_LOOKALIKE = str.maketrans(
    {"А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K", "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y", "І": "I"}
)

# Стоп-слова, которые не несут смысла в диагнозе и не должны становиться стеммами.
_STOP = frozenset(
    {
        "или", "при", "без", "для", "под", "над", "после", "перед", "другие", "других",
        "другой", "другая", "другое", "неуточненный", "неуточненная", "неуточненное",
        "неуточненные", "прочие", "прочий", "болезнь", "болезни", "синдром", "состояние",
        "состояния", "форма", "стадия", "степень", "степени", "фаза", "течение", "период",
        "года", "год", "лет", "дней", "день", "рубриках", "классифицированные",
        "классифицированных", "уточненный", "уточненная", "уточненное", "случая", "случай",
        "результате", "вследствие", "связанные", "связанная", "связанный", "включая",
        "исключая", "также", "нет", "все", "всех", "как", "что", "это", "его", "ее", "их",
    }
)

# Окончания русских слов: длинные раньше коротких. Стем не короче STEM_MIN.
_SUFFIXES = (
    "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими", "ая", "яя", "ое", "ее",
    "ые", "ие", "ой", "ый", "ий", "ых", "их", "ую", "юю", "ом", "ем", "ам", "ям", "ах", "ях",
    "ов", "ев", "ей", "ия", "ие", "ии", "ью", "а", "я", "ы", "и", "у", "ю", "о", "е", "ь", "й",
)


def normalize(text: str) -> str:
    """Нижний регистр, ё->е, пунктуация -> пробел, схлопнутые пробелы."""
    low = (text or "").lower().replace("ё", "е")
    low = re.sub(r"[^\w\s.\-]", " ", low, flags=re.UNICODE)
    low = re.sub(r"\s+", " ", low).strip()
    return low


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(normalize(text).replace(".", " ").replace("-", " ")) if t]


def stem(word: str) -> str:
    """Простой стеммер окончаний: `гипертония` -> `гипертон`, `остеохондроза` -> `остеохондроз`."""
    w = (word or "").lower().replace("ё", "е")
    if len(w) <= STEM_MIN:
        return w
    for suffix in _SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= STEM_MIN:
            return w[: -len(suffix)]
    return w


_ADJ_SUFFIXES = (
    "ого", "его", "ому", "ему", "ыми", "ими", "ая", "яя", "ое", "ее", "ые", "ой", "ый", "ий",
    "ых", "их", "ую", "юю",
)
TEXT_STEM_MIN = 5


def text_stem(word: str) -> str:
    """Стем для поиска по тексту (совпадение с начала слова).

    Короткий стем существительного дотягиваем до 5 букв, иначе «миопия» -> `миоп`
    ловила бы «миопатию». У прилагательных («острая» -> `остр`) короткий стем оставляем:
    он нужен, чтобы «острая» находила «острый»/«острое»."""
    w = (word or "").lower().replace("ё", "е")
    s = stem(w)
    if len(s) < TEXT_STEM_MIN and len(w) > TEXT_STEM_MIN and not w.endswith(_ADJ_SUFFIXES):
        return w[:TEXT_STEM_MIN]
    return s


def _pad_text(text: str) -> str:
    """Текст с границами слов для проверки «стем - начало слова»."""
    return " " + re.sub(r"[.,;()/\-]", " ", text) + " "


def _icd_prefix_from_token(token: str) -> str | None:
    """`i10` -> `I10`, `i1` -> `I1`, `i10.` -> `I10`, `i10.5` -> `I10.5`; иначе None."""
    up = (token or "").strip().upper().replace(",", ".")
    if not up or not _ICD_CODE_RE.match(up):
        return None
    return up.rstrip(".")


def _icd_prefixes_from_range(text: str) -> list[str]:
    up = (text or "").strip().upper()
    m = _ICD_RANGE_RE.match(up)
    if not m:
        return []
    letter, start, letter2, end = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    if letter2 and letter2 != letter:
        return []
    if end < start or end - start > 99:
        return []
    return [f"{letter}{n:02d}" for n in range(start, end + 1)]


@lru_cache(maxsize=1)
def _alias_file() -> dict[str, Any]:
    if not ALIASES_PATH.is_file():
        return {}
    try:
        raw = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


@lru_cache(maxsize=1)
def alias_index() -> dict[str, list[str]]:
    """Нормализованный алиас/синоним -> список фраз-расширений (без самого алиаса).

    Источники: `abbreviations` (алиас -> expand), `synonyms` (группы взаимозаменяемых
    терминов: каждый термин раскрывается в остальные).
    """
    out: dict[str, list[str]] = {}

    def add(key: str, values: Iterable[str]) -> None:
        k = normalize(key)
        if not k:
            return
        bucket = out.setdefault(k, [])
        for v in values:
            nv = normalize(v)
            if nv and nv != k and nv not in bucket:
                bucket.append(nv)

    raw = _alias_file()
    for row in raw.get("abbreviations") or []:
        if isinstance(row, dict) and row.get("alias") and row.get("expand"):
            add(str(row["alias"]), [str(row["expand"])])
    for group in raw.get("synonyms") or []:
        terms = group.get("terms") if isinstance(group, dict) else group
        if not isinstance(terms, list):
            continue
        clean = [normalize(t) for t in terms if normalize(t)]
        for term in clean:
            add(term, [t for t in clean if t != term])
    return out


@lru_cache(maxsize=1)
def alias_seed_codes() -> dict[str, list[str]]:
    """Нормализованный алиас -> seed-коды МКБ из файла (мягкая подсказка)."""
    out: dict[str, list[str]] = {}
    raw = _alias_file()
    for row in raw.get("abbreviations") or []:
        if isinstance(row, dict) and row.get("alias"):
            codes = [str(c) for c in ([row.get("seed_code")] if row.get("seed_code") else row.get("seed_codes") or []) if c]
            if codes:
                out[normalize(str(row["alias"]))] = codes
    for group in raw.get("synonyms") or []:
        if not isinstance(group, dict):
            continue
        codes = [str(c) for c in group.get("seed_codes") or [] if c]
        if not codes:
            continue
        for term in group.get("terms") or []:
            out.setdefault(normalize(str(term)), codes)
    return out


def alias_pairs_count() -> int:
    """Сколько направленных пар «алиас -> расширение» даёт словарь (для теста порога плана)."""
    return sum(len(v) for v in alias_index().values())


@lru_cache(maxsize=1)
def _icd_titles() -> tuple[tuple[str, str, frozenset[str]], ...]:
    """(код, нормализованное название без кода, множество стемм названия)."""
    if not ICD_RU_PATH.is_file():
        return ()
    try:
        rows = json.loads(ICD_RU_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    out: list[tuple[str, str, frozenset[str]]] = []
    for row in rows if isinstance(rows, list) else []:
        code = str(row.get("code") or "").strip().upper()
        title = normalize(str(row.get("title_ru") or ""))
        title = re.sub(r"^[a-z]\d{2}(?:\.\d+)?\s*-\s*", "", title)
        if not code or not title:
            continue
        stems = frozenset(stem(t) for t in tokens(title) if len(t) >= 3 and t not in _STOP)
        out.append((code, title, stems))
    return tuple(out)


@lru_cache(maxsize=1)
def icd_vocabulary() -> tuple[str, ...]:
    """Слова названий МКБ (>= 4 букв, без стоп-слов) для триграммных подсказок."""
    words: set[str] = set()
    for _code, title, _stems in _icd_titles():
        for t in tokens(title):
            if len(t) >= 4 and t not in _STOP and not t.isdigit():
                words.add(t)
    for key in alias_index():
        for t in tokens(key):
            if len(t) >= 4:
                words.add(t)
    return tuple(sorted(words))


def _trigrams(word: str) -> set[str]:
    padded = f"  {word} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


@lru_cache(maxsize=1)
def _vocab_trigrams() -> dict[str, set[str]]:
    return {w: _trigrams(w) for w in icd_vocabulary()}


def fuzzy_candidates(
    token: str,
    *,
    extra_vocab: Iterable[str] = (),
    limit: int = MAX_FUZZY,
    min_score: float = FUZZY_MIN_SCORE,
) -> list[dict[str, Any]]:
    """Ближайшие слова словаря по коэффициенту Дайса на триграммах (>= min_score)."""
    tok = normalize(token)
    if len(tok) < FUZZY_MIN_TOKEN or tok.isdigit():
        return []
    tg = _trigrams(tok)
    scored: list[tuple[float, str]] = []
    vocab = dict(_vocab_trigrams())
    for w in extra_vocab:
        nw = normalize(w)
        if nw and nw not in vocab:
            vocab[nw] = _trigrams(nw)
    for word, wtg in vocab.items():
        if word == tok:
            return []
        if abs(len(word) - len(tok)) > 3:
            continue
        inter = len(tg & wtg)
        if not inter:
            continue
        score = 2.0 * inter / (len(tg) + len(wtg))
        if score >= min_score:
            scored.append((score, word))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [{"token": tok, "suggestion": w, "score": round(s, 3)} for s, w in scored[:limit]]


def _collapse_codes(codes: Iterable[str], limit: int = MAX_CODE_PREFIXES) -> list[str]:
    """Свернуть набор кодов до префиксов: сначала полные, при переполнении - блоки из 3 знаков."""
    full = sorted({c for c in codes if c})
    if len(full) <= limit:
        return full
    blocks = sorted({c[:3] for c in full})
    if len(blocks) <= limit:
        return blocks
    return blocks[:limit]


def codes_for_phrase(phrase: str) -> list[str]:
    """Коды МКБ, в названии которых есть все стеммы фразы (стоп-слова не считаются)."""
    stems = [stem(t) for t in tokens(phrase) if len(t) >= 3 and t not in _STOP]
    stems = [s for s in stems if len(s) >= 3]
    if not stems:
        return []
    out: list[str] = []
    for code, _title, title_stems in _icd_titles():
        if "-" in code:
            continue  # блоки вида J00-J06 не бывают кодом диагноза
        if all(any(_stem_hit(ts, s) for ts in title_stems) for s in stems):
            out.append(code)
    return out


def _stem_hit(title_stem: str, query_stem: str) -> bool:
    """Стем названия совпадает со стеммой запроса; префикс - только для длинных стемм
    (`гипертон` -> `гипертоническ`), иначе `миоп` цеплял бы `миопатию`."""
    if title_stem == query_stem:
        return True
    return len(query_stem) >= 6 and title_stem.startswith(query_stem)


@dataclass
class SearchPlan:
    raw: str
    normalized: str
    tokens: list[str] = field(default_factory=list)
    icd_prefixes: list[str] = field(default_factory=list)
    phrase_stems: list[str] = field(default_factory=list)
    abbrevs: list[str] = field(default_factory=list)  # короткие сокращения целым словом: «аг», «сд»
    synonyms: list[str] = field(default_factory=list)
    synonym_stems: list[list[str]] = field(default_factory=list)
    term_codes: list[str] = field(default_factory=list)
    fuzzy: list[dict[str, Any]] = field(default_factory=list)
    fuzzy_stems: list[list[str]] = field(default_factory=list)
    doctor: bool = True
    disabled: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.normalized

    def enabled(self, chip: str) -> bool:
        return chip not in self.disabled

    def to_dict(self) -> dict[str, Any]:
        return {
            "q": self.raw,
            "normalized": self.normalized,
            "tokens": list(self.tokens),
            "chips": [
                {
                    "id": CHIP_ICD,
                    "label": "МКБ " + _range_label(self.icd_prefixes),
                    "values": list(self.icd_prefixes),
                    "enabled": self.enabled(CHIP_ICD),
                }
                if self.icd_prefixes
                else None,
                {
                    "id": CHIP_PHRASE,
                    "label": "слова "
                    + " ".join([*(f"«{s}*»" for s in self.phrase_stems), *(f"«{a}»" for a in self.abbrevs)]),
                    "values": [*self.phrase_stems, *self.abbrevs],
                    "enabled": self.enabled(CHIP_PHRASE),
                }
                if (self.phrase_stems or self.abbrevs)
                else None,
                {
                    "id": CHIP_SYNONYMS,
                    "label": "синонимы: " + ", ".join(self.synonyms),
                    "values": list(self.synonyms),
                    "enabled": self.enabled(CHIP_SYNONYMS),
                }
                if self.synonyms
                else None,
                {
                    "id": CHIP_TERMS,
                    "label": "коды по названию МКБ " + _range_label(self.term_codes),
                    "values": list(self.term_codes),
                    "enabled": self.enabled(CHIP_TERMS),
                }
                if self.term_codes
                else None,
                {
                    "id": CHIP_FUZZY,
                    "label": "похожие слова: " + ", ".join(f["suggestion"] for f in self.fuzzy),
                    "values": [f["suggestion"] for f in self.fuzzy],
                    "enabled": self.enabled(CHIP_FUZZY),
                }
                if self.fuzzy
                else None,
                {
                    "id": CHIP_DOCTOR,
                    "label": f"врач или ID «{self.normalized}»",
                    "values": [self.normalized],
                    "enabled": self.enabled(CHIP_DOCTOR),
                }
                if self.has_doctor_chip
                else None,
            ],
            "disabled": list(self.disabled),
        }

    @property
    def has_doctor_chip(self) -> bool:
        return bool(self.doctor and self.normalized and not self.icd_prefixes)


def _range_label(codes: list[str]) -> str:
    if not codes:
        return ""
    if len(codes) == 1:
        return codes[0]
    if len(codes) <= 3:
        return ", ".join(codes)
    return f"{codes[0]}-{codes[-1]} ({len(codes)})"


def _parse_disabled(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple, set)):
        items = [str(x) for x in raw]
    else:
        items = re.split(r"[|,;\s]+", str(raw))
    return [x.strip().lower() for x in items if x.strip().lower() in ALL_CHIPS]


def expand_query(q: str, *, disabled: Any = None, extra_vocab: Iterable[str] = ()) -> SearchPlan:
    """Построить план поиска по строке запроса. Пустой запрос -> пустой план."""
    raw = (q or "").strip()
    # «хр. панкреатит» -> «хронический панкреатит» (word_expansions словаря).
    expanded_raw = _apply_word_expansions(raw, _word_expansions())
    normalized = normalize(expanded_raw)
    plan = SearchPlan(raw=raw, normalized=normalized, disabled=_parse_disabled(disabled))
    if not normalized:
        return plan

    # Коды и диапазоны МКБ.
    prefixes: list[str] = []
    for part in re.split(r"[\s,;]+", raw.strip()):
        latin = part.strip().upper().translate(_CYR_TO_LATIN_LOOKALIKE)
        rng = _icd_prefixes_from_range(latin)
        if rng:
            prefixes.extend(rng)
            continue
        pref = _icd_prefix_from_token(latin)
        if pref:
            prefixes.append(pref)
    whole_range = _icd_prefixes_from_range(raw.strip().upper().translate(_CYR_TO_LATIN_LOOKALIKE))
    if whole_range:
        prefixes = whole_range
    plan.icd_prefixes = _collapse_codes(prefixes)

    # Слова запроса (без кодов).
    words = [
        t
        for t in tokens(normalized)
        if not _icd_prefix_from_token(t.upper().translate(_CYR_TO_LATIN_LOOKALIKE)) and not t.isdigit()
    ]
    plan.tokens = words
    index = alias_index()
    meaningful = [w for w in words if len(w) >= 3 and w not in _STOP]
    plan.phrase_stems = _dedupe([text_stem(w) for w in meaningful if len(stem(w)) >= 3])
    # Двухбуквенные сокращения из словаря («аг», «сд») ищутся целым словом, не подстрокой.
    plan.abbrevs = _dedupe([w for w in words if len(w) < 3 and w in index])

    # Алиасы: весь запрос целиком, затем отдельные слова.
    expansions: list[str] = []
    if normalized in index:
        expansions.extend(index[normalized])
    for w in [*meaningful, *plan.abbrevs]:
        for exp in index.get(w, []):
            if exp not in expansions and exp != normalized:
                expansions.append(exp)
    expansions = expansions[:MAX_PHRASES]
    plan.synonyms = expansions
    plan.synonym_stems = [
        _dedupe([text_stem(t) for t in tokens(exp) if len(t) >= 3 and t not in _STOP and len(stem(t)) >= 3])
        for exp in expansions
    ]

    # Термин -> коды МКБ по названиям: сам запрос, seed-коды алиасов и только многословные
    # синонимы (однословный «рефлюкс» тянул бы рефлюкс-уропатию N11.0; текстом он и так ищется).
    if meaningful or plan.abbrevs:
        codes: set[str] = set()
        if meaningful:
            codes.update(codes_for_phrase(normalized))
        for exp, stems in zip(expansions, plan.synonym_stems):
            if len(stems) >= 2:
                codes.update(codes_for_phrase(exp))
        for key in [normalized, *meaningful, *plan.abbrevs]:
            codes.update(alias_seed_codes().get(key, []))
        plan.term_codes = _collapse_codes(codes)

    # Опечатки: только для слов, которых нет ни в словаре, ни среди алиасов.
    # Проверка пословная: в «острая респираторнная» опечатка второго слова исправляется,
    # хотя первое слово уже дало коды. Слово из словаря («гипертония») похожих не получает -
    # иначе оно тянуло бы «гипотонию».
    vocab = set(icd_vocabulary())
    fuzzy: list[dict[str, Any]] = []
    for w in meaningful:
        if w in vocab or w in index:
            continue
        if any(w == t for exp in expansions for t in tokens(exp)):
            continue
        fuzzy.extend(fuzzy_candidates(w, extra_vocab=extra_vocab))
    unique_fuzzy: list[dict[str, Any]] = []
    seen_stems: set[str] = set()
    for item in fuzzy:
        st = stem(item["suggestion"])
        if len(st) < 3 or st in seen_stems:
            continue
        seen_stems.add(st)
        unique_fuzzy.append(item)
    plan.fuzzy = unique_fuzzy[:MAX_FUZZY]
    plan.fuzzy_stems = [[text_stem(f["suggestion"])] for f in plan.fuzzy]
    return plan


@lru_cache(maxsize=1)
def _word_expansions() -> list[dict[str, Any]]:
    rows = _alias_file().get("word_expansions")
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _like_contains(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _case_variants(value: str) -> list[str]:
    """LOWER() в SQLite не трогает кириллицу, поэтому регистр закрываем вариантами:
    `гипертон` / `Гипертон` / `ГИПЕРТОН` - строчный, с заглавной, капсом."""
    low = value.lower()
    out = [low]
    for variant in (low[:1].upper() + low[1:], low.upper()):
        if variant not in out:
            out.append(variant)
    return out


def _ci_like(column: str, value: str) -> tuple[str, list[str]]:
    """`column` содержит `value` без учёта регистра кириллицы и с ё=е."""
    col = f"REPLACE(REPLACE(COALESCE({column}, ''), 'ё', 'е'), 'Ё', 'Е')"
    variants = _case_variants(value)
    clause = "(" + " OR ".join(f"{col} LIKE ? ESCAPE '\\'" for _ in variants) + ")"
    return clause, [_like_contains(v) for v in variants]


def _padded_col(column: str) -> str:
    """Колонка с ё=е и границами слов (знаки -> пробел, пробелы по краям)."""
    inner = f"REPLACE(REPLACE(COALESCE({column}, ''), 'ё', 'е'), 'Ё', 'Е')"
    for ch in (",", ".", ";", "(", ")", "/", "-"):
        inner = f"REPLACE({inner}, '{ch}', ' ')"
    return f"(' ' || {inner} || ' ')"


def _ci_word(column: str, word: str) -> tuple[str, list[str]]:
    """`column` содержит `word` целым словом (границы - пробел/знаки), без учёта регистра."""
    col = _padded_col(column)
    variants = _case_variants(word)
    clause = "(" + " OR ".join(f"{col} LIKE ? ESCAPE '\\'" for _ in variants) + ")"
    return clause, [f"% {v} %" for v in variants]


def _ci_prefix(column: str, stem_value: str) -> tuple[str, list[str]]:
    """В `column` есть слово, начинающееся со `stem_value` (не подстрока внутри слова)."""
    col = _padded_col(column)
    variants = _case_variants(stem_value)
    clause = "(" + " OR ".join(f"{col} LIKE ? ESCAPE '\\'" for _ in variants) + ")"
    return clause, [f"% {_like_contains(v)[1:]}" for v in variants]


def _stems_clause(stems: list[str], columns: list[str], abbrevs: list[str] | None = None) -> tuple[str, list[str]]:
    """Все стеммы (и сокращения целым словом) должны встретиться в одной колонке."""
    parts: list[str] = []
    values: list[str] = []
    for column in columns:
        ands = []
        for s in stems:
            clause, vals = _ci_prefix(column, s)
            ands.append(clause)
            values.extend(vals)
        for a in abbrevs or []:
            clause, vals = _ci_word(column, a)
            ands.append(clause)
            values.extend(vals)
        parts.append("(" + " AND ".join(ands) + ")")
    return "(" + " OR ".join(parts) + ")", values


def sql_parts(plan: SearchPlan) -> dict[str, tuple[str, list[Any]]]:
    """SQL-фрагменты по чипам: id -> (clause, values). Пустые/выключенные чипы отсутствуют.

    Алиасы: `c` - fact_mo_case, `d` - dim_doctor, `dx` - dim_diagnosis.
    """
    text_cols = ["c.diagnosis_text", "dx.diagnosis_label"]
    out: dict[str, tuple[str, list[Any]]] = {}
    if plan.icd_prefixes and plan.enabled(CHIP_ICD):
        ors = []
        values: list[Any] = []
        for pref in plan.icd_prefixes:
            ors.append("REPLACE(UPPER(COALESCE(c.diagnosis_code, '')), '.', '') LIKE ? ESCAPE '\\'")
            values.append(pref.replace(".", "") + "%")
        out[CHIP_ICD] = ("(" + " OR ".join(ors) + ")", values)
    if (plan.phrase_stems or plan.abbrevs) and plan.enabled(CHIP_PHRASE):
        out[CHIP_PHRASE] = _stems_clause(plan.phrase_stems, text_cols, plan.abbrevs)
    if plan.synonym_stems and plan.enabled(CHIP_SYNONYMS):
        ors = []
        values = []
        for stems in plan.synonym_stems:
            if not stems:
                continue
            clause, vals = _stems_clause(stems, text_cols)
            ors.append(clause)
            values.extend(vals)
        if ors:
            out[CHIP_SYNONYMS] = ("(" + " OR ".join(ors) + ")", values)
    if plan.term_codes and plan.enabled(CHIP_TERMS):
        ors = []
        values = []
        for code in plan.term_codes:
            ors.append("REPLACE(UPPER(COALESCE(c.diagnosis_code, '')), '.', '') LIKE ? ESCAPE '\\'")
            values.append(code.replace(".", "") + "%")
        out[CHIP_TERMS] = ("(" + " OR ".join(ors) + ")", values)
    if plan.fuzzy_stems and plan.enabled(CHIP_FUZZY):
        ors = []
        values = []
        for stems in plan.fuzzy_stems:
            clause, vals = _stems_clause(stems, text_cols)
            ors.append(clause)
            values.extend(vals)
        out[CHIP_FUZZY] = ("(" + " OR ".join(ors) + ")", values)
    if plan.has_doctor_chip and plan.enabled(CHIP_DOCTOR):
        fio_clause, fio_values = _ci_like("d.doctor_fio", plan.normalized)
        doctor_clause = (
            f"({fio_clause} OR CAST(c.visit_id AS TEXT) LIKE ? ESCAPE '\\' "
            "OR CAST(c.mis_id AS TEXT) LIKE ? ESCAPE '\\')"
        )
        out[CHIP_DOCTOR] = (doctor_clause, [*fio_values, _like_contains(plan.normalized), _like_contains(plan.normalized)])
    return out


def sql_clause(plan: SearchPlan) -> tuple[str, list[Any]]:
    """Итоговый WHERE-фрагмент: OR по всем включённым чипам. Пустой план -> ('1=0', [])."""
    parts = sql_parts(plan)
    if not parts:
        return ("1=0", [])
    clause = "(" + " OR ".join(c for c, _v in parts.values()) + ")"
    values: list[Any] = []
    for _c, v in parts.values():
        values.extend(v)
    return clause, values


def sql_rank(plan: SearchPlan) -> tuple[str, list[Any]]:
    """CASE-ранг: код (1) > фраза (2) > синонимы (3) > коды по названию (4) > похожие (5) > врач/ID (6)."""
    parts = sql_parts(plan)
    if not parts:
        return ("7", [])
    whens: list[str] = []
    values: list[Any] = []
    for rank, chip in enumerate(_RANK_ORDER, start=1):
        if chip in parts:
            clause, vals = parts[chip]
            whens.append(f"WHEN {clause} THEN {rank}")
            values.extend(vals)
    return ("CASE " + " ".join(whens) + " ELSE 7 END", values)


_RANK_ORDER = (CHIP_ICD, CHIP_PHRASE, CHIP_SYNONYMS, CHIP_TERMS, CHIP_FUZZY, CHIP_DOCTOR)
_SAFE_LITERAL_RE = re.compile(r"[^0-9a-zA-Zа-яА-ЯёЁ .\-%]")


def chip_for_rank(rank: int) -> str | None:
    """1..6 -> id чипа; 0/7 -> None."""
    if 1 <= rank <= len(_RANK_ORDER):
        return _RANK_ORDER[rank - 1]
    return None


def _inline_literal(value: Any) -> str:
    """Литерал для CASE-ранга: только буквы/цифры/пробел/точка/дефис/%, в одинарных кавычках.

    Значения плана - нормализованные стеммы, префиксы кодов и LIKE-шаблоны, поэтому
    фильтр символов ничего осмысленного не теряет, а инъекцию исключает.
    """
    text = _SAFE_LITERAL_RE.sub("", str(value))
    return "'" + text + "'"


def sql_rank_inline(plan: SearchPlan) -> str:
    """Тот же ранг, что `sql_rank`, но без параметров: удобно ставить в ORDER BY и GROUP BY,
    где список `?` пришлось бы дублировать в двух местах запроса."""
    parts = sql_parts(plan)
    if not parts:
        return "7"
    whens: list[str] = []
    for rank, chip in enumerate(_RANK_ORDER, start=1):
        if chip not in parts:
            continue
        clause, vals = parts[chip]
        pieces = clause.split("?")
        assert len(pieces) == len(vals) + 1
        inlined = pieces[0]
        for piece, val in zip(pieces[1:], vals):
            inlined += _inline_literal(val) + piece
        whens.append(f"WHEN {inlined} THEN {rank}")
    return "CASE " + " ".join(whens) + " ELSE 7 END"


def _rec_text(rec: dict[str, Any]) -> str:
    return normalize(
        " ".join(
            str(rec.get(k) or "")
            for k in ("diagnosis_text", "diagnosis_short", "diagnosis_label")
        )
    )


def _rec_code(rec: dict[str, Any]) -> str:
    return str(rec.get("diagnosis_code") or rec.get("mkb_code_main") or "").upper().replace(".", "")


def match_record(plan: SearchPlan, rec: dict[str, Any]) -> int | None:
    """Ранг совпадения записи (как sql_rank) или None, если не найдено. Для JSONL-пути."""
    if plan.is_empty:
        return 0
    code = _rec_code(rec)
    text = _rec_text(rec)
    if plan.icd_prefixes and plan.enabled(CHIP_ICD):
        if any(code.startswith(p.replace(".", "")) for p in plan.icd_prefixes):
            return 1
    padded = _pad_text(text)
    if (plan.phrase_stems or plan.abbrevs) and plan.enabled(CHIP_PHRASE):
        if all(f" {s}" in padded for s in plan.phrase_stems) and all(f" {a} " in padded for a in plan.abbrevs):
            return 2
    if plan.synonym_stems and plan.enabled(CHIP_SYNONYMS):
        if any(stems and all(f" {s}" in padded for s in stems) for stems in plan.synonym_stems):
            return 3
    if plan.term_codes and plan.enabled(CHIP_TERMS):
        if any(code.startswith(c.replace(".", "")) for c in plan.term_codes):
            return 4
    if plan.fuzzy_stems and plan.enabled(CHIP_FUZZY):
        if any(all(f" {s}" in padded for s in stems) for stems in plan.fuzzy_stems):
            return 5
    if plan.has_doctor_chip and plan.enabled(CHIP_DOCTOR):
        hay = normalize(
            " ".join(str(rec.get(k) or "") for k in ("doctor_fio", "visit_id", "case_id", "mis_id"))
        )
        if plan.normalized in hay:
            return 6
    return None


def suggest(q: str, *, limit: int = 8, extra_labels: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Автодополнение: алиасы/синонимы, слова словаря МКБ, названия из склада, похожие слова."""
    norm = normalize(q)
    if len(norm) < 2:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def push(label: str, kind: str, value: str | None = None) -> None:
        key = normalize(label)
        if not key or key in seen or len(out) >= limit:
            return
        seen.add(key)
        out.append({"label": label, "kind": kind, "value": value or label})

    index = alias_index()
    for key in sorted(index):
        if key.startswith(norm):
            push(key, "alias")
        elif norm in key:
            push(key, "alias")
    for key, expansions in index.items():
        if key == norm:
            for exp in expansions:
                push(exp, "synonym")
    for label in extra_labels:
        if norm in normalize(label):
            push(str(label), "term")
    last = norm.split(" ")[-1]
    if len(last) >= 3:
        for word in icd_vocabulary():
            if word.startswith(last) and word != last:
                push(word, "word")
                if len(out) >= limit:
                    break
    if len(out) < limit:
        # Пока слово печатается, порог ниже: «остеохандр» -> «остеохондроз».
        for cand in fuzzy_candidates(last, min_score=0.5):
            push(cand["suggestion"], "typo")
    return out[:limit]


def clear_cache() -> None:
    for fn in (_alias_file, alias_index, alias_seed_codes, _icd_titles, icd_vocabulary, _vocab_trigrams):
        fn.cache_clear()
