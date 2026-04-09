"""
МКБ-10: извлечение кодов из текста, лексическое сопоставление ТОЛЬКО с русским справочником
(один JSON, собирается из официального Excel — см. scripts/export_icd_ru_from_xlsx.py).
Англ. названия WHO (опционально) — только для подписи к уже известному коду, не для лексикона.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ICD-10: первая буква категории + две цифры + необязательный подуровень (без U).
ICD10_CODE_RE = re.compile(
    r"\b([A-TV-Z]\d{2}(?:\.\d{1,4})?)\b",
    re.IGNORECASE,
)
ICD10_TERMINAL_RU_RE = re.compile(r"^[A-TV-Z]\d{2}(?:\.\d{1,4})?$", re.IGNORECASE)

# Слишком общие слова для лексического матчинга по названию МКБ.
# «Хронический/острый» не отфильтровываем: в рубриках J** они различают диагнозы (J01 vs J32).
_RU_STOP = frozenset(
    """
    болезнь болезни заболевание заболевания диагноз код мкб мкб-10 жалоба жалобы
    пациент пациентка симптом симптомы
    для при или без над под про как что это все всех между через
    дифференциальная гипотеза подбора протокола протокол протокола
    """.split()
)

# Сами по себе слабо различают рубрику; в паре с анатомией/носологией полезны.
_RU_WEAK_ADJ = frozenset(
    """
    хронический хроническая хроническое хронические
    острый острая острое острые
    подострый подострая
    """.split()
)


def _norm_icd_code(s: str) -> str:
    s = (s or "").strip().upper().replace(",", ".").replace(" ", "")
    if s.endswith(".-"):
        s = s[:-2]
    return s


def normalize_icd_code(s: str) -> str:
    """Публичная нормализация кода МКБ-10 для сравнения и валидации."""
    return _norm_icd_code(s)


@lru_cache(maxsize=1)
def _load_who_rows() -> list[dict]:
    p = ROOT / "data/icd_reference/icd10_who_2016_terminal_codes.json"
    if not p.is_file():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _who_by_code() -> dict[str, dict]:
    m: dict[str, dict] = {}
    for row in _load_who_rows():
        c = (row.get("code") or "").strip().upper()
        if not c:
            continue
        m[c] = row
        n = _norm_icd_code(c)
        if n and n not in m:
            m[n] = row
    return m


def _ru_json_path() -> Path:
    rel = (os.environ.get("ICD_RU_JSON") or "").strip()
    if rel:
        pp = Path(rel)
        return pp if pp.is_absolute() else (ROOT / pp)
    return ROOT / "data/icd_reference/icd10_ru_mkb10su.json"


@lru_cache(maxsize=1)
def _ru_rows() -> list[dict]:
    p = _ru_json_path()
    if not p.is_file():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _ru_valid_codes() -> frozenset[str]:
    """Коды из единственного русского справочника (Excel → JSON)."""
    return frozenset(
        _norm_icd_code(str(r.get("code") or ""))
        for r in _ru_rows()
        if (r.get("code") or "").strip()
    )


def is_code_in_ru_reference(code: str) -> bool:
    c = _norm_icd_code(code)
    return bool(c) and c in _ru_valid_codes()


def who_title_en(code: str) -> str | None:
    c = _norm_icd_code(code)
    wb = _who_by_code()
    row = wb.get(c) or wb.get(code.upper())
    if not row:
        if len(c) == 3 and c[0].isalpha() and c[1:].isdigit():
            row = wb.get(c + ".-")
    if not row:
        return None
    return (row.get("title_en") or "").strip() or None


def ru_title(code: str) -> str | None:
    c = _norm_icd_code(code)
    for row in _ru_rows():
        if _norm_icd_code(row.get("code") or "") == c:
            t = (row.get("title_ru") or "").strip()
            return t or None
    return None


def extract_icd_codes_raw(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in ICD10_CODE_RE.finditer(text):
        raw = m.group(1)
        n = _norm_icd_code(raw)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def count_icd_code_mentions(text: str, *, top_n: int = 5) -> list[dict]:
    """Топ кодов МКБ-10 по числу вхождений; только коды из русского справочника."""
    if not text or top_n <= 0:
        return []
    counts: Counter[str] = Counter()
    for m in ICD10_CODE_RE.finditer(text):
        n = _norm_icd_code(m.group(1))
        if not n or not is_code_in_ru_reference(n):
            continue
        counts[n] += 1
    if not counts:
        return []
    out: list[dict] = []
    for code, cnt in counts.most_common(top_n):
        tru = ru_title(code) or ""
        out.append({"code": code, "count": int(cnt), "title_ru": tru})
    return out


def icd_tokens_for_lex(codes: list[str]) -> set[str]:
    """Токены для лексического RAG (латиница+цифры), чтобы «J20.9» давал j20, j20.9."""
    out: set[str] = set()
    for c in codes:
        n = _norm_icd_code(c)
        if not n:
            continue
        low = n.lower()
        out.add(low)
        if "." in low:
            base, _, rest = low.partition(".")
            out.add(base)
            out.add(base + rest.replace(".", ""))
        else:
            out.add(low)
    return {t for t in out if len(t) >= 2}


def describe_code(code: str) -> dict:
    c = _norm_icd_code(code)
    ten = who_title_en(c)
    tru = ru_title(c)
    return {
        "code": c,
        "title_ru": tru,
        "title_en": ten,
    }


def resolve_extracted_codes(codes: list[str]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for c in codes:
        n = _norm_icd_code(c)
        if not n or n in seen:
            continue
        seen.add(n)
        d = describe_code(n)
        d["match_method"] = "regex_query"
        out.append(d)
    return out


def _icd_extra_roots(text: str) -> list[str]:
    """Подсказки к МКБ: разговорное «гайморит» ↔ формулировки справочника («верхнечелюстной»)."""
    s = text.lower().replace("ё", "е")
    extra: list[str] = []
    if "гаймор" in s or "гайморит" in s:
        extra.extend(["верхнечелюстн", "верхнечелюстной", "гайморова"])
    if "риносинусит" in s or "синусит" in s:
        extra.append("синусит")
    return extra


def _ru_words(text: str) -> list[str]:
    s = text.lower().replace("ё", "е")
    words = [w for w in re.findall(r"[а-яё]{3,}", s) if w not in _RU_STOP]
    for r in _icd_extra_roots(s):
        if r not in words and len(r) >= 3:
            words.append(r)
    return words


def _lexicon_score_one_row(
    words: list[str], qlow: str, code: str, title: str
) -> float:
    tlow = title.lower().replace("ё", "е")
    score = 0.0
    for w in words:
        if w not in tlow:
            continue
        if w in _RU_WEAK_ADJ:
            score += 1.0
            continue
        if len(w) >= 5:
            score += 3.0
        elif len(w) == 4:
            score += 2.0
        else:
            score += 1.0
    if len(qlow) >= 6 and qlow in tlow:
        score += 8.0
    if words and len(tlow) <= 48:
        hit = sum(1 for w in words if w in tlow)
        if hit >= 2:
            score += 2.5
    if len(tlow) > 55:
        score -= min(4.0, (len(tlow) - 55) * 0.06)
    if score <= 0:
        return 0.0
    # Раньше: +1.8 ко всем *.9 без контекста — тянуло «неуточнённ» коды из-за частицы «для» и т.п.
    if code.endswith(".9") and score >= 4.5 and not any(
        x in qlow
        for x in ("mycoplasma", "микоплазм", "вирус", "бактер", "стрептококк", "гемофил")
    ):
        score += 1.2
    return score


def ru_lexicon_scored_entries(text: str) -> list[dict]:
    """
    Все коды с положительным лексическим score, по убыванию score;
    на каждый код — строка с максимальным score (агрегация по коду).
    """
    if not text or len(text.strip()) < 3:
        return []
    words = _ru_words(text)
    qlow = text.lower().replace("ё", "е").strip()
    best: dict[str, tuple[float, str, str]] = {}
    for row in _ru_rows():
        code = (row.get("code") or "").strip()
        if not ICD10_TERMINAL_RU_RE.match(code):
            continue
        title = (row.get("title_ru") or "").strip()
        if not title:
            continue
        sc = _lexicon_score_one_row(words, qlow, code, title)
        if sc <= 0:
            continue
        n = _norm_icd_code(code)
        prev = best.get(n)
        if prev is None or sc > prev[0]:
            best[n] = (sc, n, title)
    out: list[dict] = []
    for sc, code, title in sorted(best.values(), key=lambda x: -x[0]):
        ten = who_title_en(code)
        out.append(
            {
                "code": code,
                "title_ru": title,
                "title_en": ten,
                "lex_score": round(sc, 2),
            }
        )
    return out


def suggest_icd_from_russian(text: str, max_results: int = 8) -> list[dict]:
    """Лексическое сопоставление запроса с русскими названиями МКБ (без LLM)."""
    if not text or len(text.strip()) < 3:
        return []
    words = _ru_words(text)
    qlow = text.lower().replace("ё", "е").strip()
    scored: list[tuple[float, str, str]] = []
    for row in _ru_rows():
        code = (row.get("code") or "").strip()
        if not ICD10_TERMINAL_RU_RE.match(code):
            continue
        title = (row.get("title_ru") or "").strip()
        if not title:
            continue
        sc = _lexicon_score_one_row(words, qlow, code, title)
        if sc <= 0:
            continue
        scored.append((sc, code, title))
    scored.sort(key=lambda x: -x[0])

    def _stem(c: str) -> str:
        c = _norm_icd_code(c)
        if len(c) >= 3 and c[0].isalpha() and c[1:3].isdigit():
            return c[:3].upper()
        return c

    out: list[dict] = []
    seen: set[str] = set()
    stems_used: set[str] = set()
    for score, code, title in scored:
        n = _norm_icd_code(code)
        if n in seen:
            continue
        st = _stem(n)
        if st in stems_used:
            continue
        seen.add(n)
        stems_used.add(st)
        ten = who_title_en(n)
        out.append(
            {
                "code": n,
                "title_ru": title,
                "title_en": ten,
                "match_method": "lexicon_ru",
                "score": round(score, 2),
            }
        )
        if len(out) >= max_results:
            break
    return out


def analyze_query_for_icd(full_query: str, rag_query: str) -> dict:
    """
    Объединяет: коды из полного запроса и RAG-части + лексические гипотезы по русскому тексту.

    Если в тексте явно указаны коды МКБ — используются только они (после проверки по русскому
    справочнику из Excel); лексические гипотезы не подмешиваются.
    """
    combined = f"{full_query}\n{rag_query}"
    extracted = extract_icd_codes_raw(combined)
    detected_raw = resolve_extracted_codes(extracted)

    detected_valid: list[dict] = []
    detected_unknown: list[dict] = []
    for d in detected_raw:
        c = normalize_icd_code(str(d.get("code") or ""))
        if not c:
            continue
        if is_code_in_ru_reference(c):
            detected_valid.append(d)
        else:
            detected_unknown.append({**d, "code": c, "match_method": "regex_query_unknown"})

    # Явные коды в тексте, но ни один не из справочника — откатываемся к лексикону.
    explicit_codes_in_query = bool(detected_raw)
    if explicit_codes_in_query and not detected_valid and detected_unknown:
        explicit_codes_in_query = False

    suggested: list[dict] = []
    if rag_query.strip() and not explicit_codes_in_query:
        suggested = suggest_icd_from_russian(rag_query, max_results=8)

    det_set = {d["code"] for d in detected_valid}
    suggested = [s for s in suggested if s["code"] not in det_set]

    if explicit_codes_in_query:
        codes_for_retrieval = [d["code"] for d in detected_valid][:10]
    else:
        codes_for_retrieval = list(
            dict.fromkeys(
                [d["code"] for d in detected_valid]
                + [s["code"] for s in suggested[:8]]
            )
        )[:10]

    return {
        "detected": detected_valid,
        "detected_unknown": detected_unknown,
        "suggested": suggested,
        "codes_for_retrieval": codes_for_retrieval,
        "explicit_icd_in_query": explicit_codes_in_query,
    }
