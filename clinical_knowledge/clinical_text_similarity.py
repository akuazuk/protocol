"""Общая нормализация и сходство клинических формулировок (stdlib).

Потребители:
- mo_icd_name_match: диагноз ↔ title_ru справочника МКБ (без кодов);
- позже: диагноз ↔ жалобы / анамнез / обследования / рекомендации / лечение.

Не смешивать с проверкой «код есть в справочнике» (mo_icd_directory_eval).

Лёгкий stem (фаза 6 ICD pipeline): ``MO_ICD_LIGHT_STEM=1`` - отрезание 1-2
окончаний у токенов ≥6; без pymorphy. Default off.
"""
from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from typing import Any

# МКБ-10 латиница (A00, J20.9, Z00.0)
_ICD_CODE_RE = re.compile(r"\b[A-Za-z]\d{2}(?:\.\d{1,4})?\b")
# Ведущий «A00.0 - » / «A00.0:» в title_ru выгрузки
_LEADING_CODE_TITLE_RE = re.compile(
    r"^\s*[A-Za-z]\d{2}(?:\.\d{1,4})?\s*[-- - :]\s*",
    re.UNICODE,
)
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]{3,}", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\sа-яёА-ЯЁ]", re.UNICODE)

# Только 1-2 символа (план v3 §4.4). Более длинные - сначала, stem ≥4.
_LIGHT_STEM_ENDINGS: tuple[str, ...] = (
    "ов",
    "ев",
    "ам",
    "ям",
    "ом",
    "ем",
    "ах",
    "ях",
    "ой",
    "ей",
    "ий",
    "ый",
    "ое",
    "ее",
    "ая",
    "яя",
    "ые",
    "ие",
    "а",
    "я",
    "у",
    "ю",
    "е",
    "ы",
    "и",
    "о",
)


def light_stem_enabled() -> bool:
    raw = (os.environ.get("MO_ICD_LIGHT_STEM") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def light_stem(token: str) -> str:
    """Лёгкий stem без морфологического словаря (животе/живота → живот)."""
    t = (token or "").replace("ё", "е").replace("Ё", "е").lower()
    if len(t) < 6:
        return t
    for end in _LIGHT_STEM_ENDINGS:
        if t.endswith(end) and len(t) - len(end) >= 4:
            return t[: -len(end)]
    return t


def strip_icd_codes(text: str) -> str:
    """Убрать коды МКБ из строки (для name_only и секций клиники).

    Сначала нормализует кириллические/пробельные формы («М21.4», «Е 55.0»)
    через icd_mkb, затем вырезает латинские токены кода.
    """
    if not text:
        return ""
    raw = str(text)
    try:
        import icd_mkb

        raw = icd_mkb.normalize_text_for_icd_scan(raw)
    except Exception:  # noqa: BLE001
        pass
    return _WS_RE.sub(" ", _ICD_CODE_RE.sub(" ", raw)).strip()


def split_diagnosis_phrases(text: str) -> list[str]:
    """Разбить мультидиагноз на фразы для name-match.

    Длинный клинический текст с несколькими нозологиями нельзя сравнивать
    целиком с коротким title_ru - короткий title «выигрывает» overlap'ом.
    Берём предложения и «голову» до первой запятой (нозология без уточнений).
    """
    cleaned = strip_icd_codes(text or "")
    if not cleaned:
        return []
    parts = re.split(r"[.?!;\n]+", cleaned)
    out: list[str] = []
    seen: set[str] = set()

    def _add(phrase: str) -> None:
        p = _WS_RE.sub(" ", (phrase or "").strip(" ,;:-"))
        if len(p) < 4:
            return
        key = p.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    for part in parts:
        _add(part)
        # «Бронхиальная астма, аллергическая, …» → отдельно нозология
        if "," in (part or ""):
            head = part.split(",", 1)[0]
            if 4 <= len(head.strip()) <= 80:
                _add(head)
    if not out and cleaned.strip():
        _add(cleaned)
    return out


def best_combined_against_title(query: str, title: str) -> dict[str, Any]:
    """Max combined_score по фразам мультидиагноза и полному тексту."""
    title_clean = strip_leading_code_from_title(title or "")
    q = (query or "").strip()
    if not q or len(title_clean) < 3:
        return {
            "token_jaccard": 0.0,
            "token_coverage": 0.0,
            "fuzz_ratio": 0.0,
            "combined": 0.0,
            "matched_phrase": "",
        }
    phrases = split_diagnosis_phrases(q)
    if q not in phrases:
        phrases = [q, *phrases]
    best = combined_score(q, title_clean)
    best_phrase = q
    for phrase in phrases:
        scores = combined_score(phrase, title_clean)
        if float(scores["combined"]) > float(best["combined"]):
            best = scores
            best_phrase = phrase
    return {**best, "matched_phrase": best_phrase}


def strip_leading_code_from_title(title: str) -> str:
    """Снять префикс «CODE - » из title_ru справочника."""
    if not title:
        return ""
    return _LEADING_CODE_TITLE_RE.sub("", title).strip()


def normalize_for_match(text: str, *, strip_codes: bool = True) -> str:
    """Нижний регистр, ё→е, без пунктуации; опционально без кодов МКБ."""
    raw = (text or "").replace("ё", "е").replace("Ё", "е")
    if strip_codes:
        raw = strip_icd_codes(raw)
        raw = strip_leading_code_from_title(raw)
    raw = _PUNCT_RE.sub(" ", raw.lower())
    return _WS_RE.sub(" ", raw).strip()


def tokens(text: str, *, min_len: int = 3, stem: bool | None = None) -> set[str]:
    norm = normalize_for_match(text, strip_codes=True)
    raw = {t for t in _TOKEN_RE.findall(norm) if len(t) >= min_len}
    if stem is None:
        stem = light_stem_enabled()
    if not stem:
        return raw
    return {light_stem(t) for t in raw}


def token_jaccard(a: str, b: str, *, min_len: int = 3) -> float:
    ta, tb = tokens(a, min_len=min_len), tokens(b, min_len=min_len)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def token_coverage(query: str, reference: str, *, min_len: int = 3) -> float:
    """Доля токенов reference, покрытых query (ближе к title_match_score v1)."""
    tq, tr = tokens(query, min_len=min_len), tokens(reference, min_len=min_len)
    if not tq or not tr:
        return 0.0
    return len(tq & tr) / max(len(tr), 1)


def fuzz_ratio(a: str, b: str) -> float:
    na, nb = normalize_for_match(a), normalize_for_match(b)
    if not na or not nb:
        return 0.0
    return float(SequenceMatcher(None, na, nb).ratio())


def combined_score(
    a: str,
    b: str,
    *,
    jaccard_weight: float = 0.45,
    coverage_weight: float = 0.25,
    fuzz_weight: float = 0.30,
) -> dict[str, Any]:
    """Сводный score 0..1 для пары клинических строк."""
    jac = token_jaccard(a, b)
    cov = token_coverage(a, b)
    fuzz = fuzz_ratio(a, b)
    wsum = max(jaccard_weight + coverage_weight + fuzz_weight, 1e-9)
    combined = (jaccard_weight * jac + coverage_weight * cov + fuzz_weight * fuzz) / wsum
    return {
        "token_jaccard": round(jac, 4),
        "token_coverage": round(cov, 4),
        "fuzz_ratio": round(fuzz, 4),
        "combined": round(combined, 4),
    }


def score_against_sections(
    diagnosis: str,
    sections: dict[str, str],
) -> dict[str, Any]:
    """Заготовка для фазы D: Dx ↔ жалобы/анамнез/обследования/план.

    Не пишет findings - только профиль similarity по секциям.
    """
    profile: dict[str, Any] = {}
    best_name = None
    best_combined = -1.0
    for name, text in (sections or {}).items():
        if not str(text or "").strip():
            profile[name] = {"combined": 0.0, "empty": True}
            continue
        scores = combined_score(diagnosis, str(text))
        profile[name] = {**scores, "empty": False}
        if scores["combined"] > best_combined:
            best_combined = float(scores["combined"])
            best_name = name
    return {
        "by_section": profile,
        "best_section": best_name,
        "best_combined": round(best_combined, 4) if best_combined >= 0 else 0.0,
    }


def best_match_against_titles(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    title_key: str = "title_ru",
) -> dict[str, Any] | None:
    """Выбрать лучший кандидат по max score фраз мультидиагноза ↔ title.

    candidates: list of dicts with title_key (и обычно code).
    """
    q = (query or "").strip()
    if len(q) < 3 or not candidates:
        return None
    best: dict[str, Any] | None = None
    best_score = -1.0
    for row in candidates:
        if not isinstance(row, dict):
            continue
        title = strip_leading_code_from_title(str(row.get(title_key) or ""))
        if len(title) < 3:
            continue
        scores = best_combined_against_title(q, title)
        sc = float(scores["combined"])
        if sc > best_score:
            best_score = sc
            best = {
                **{k: row.get(k) for k in row},
                "title_ru_clean": title,
                "similarity": scores,
                "score": sc,
                "matched_phrase": scores.get("matched_phrase") or "",
            }
    return best
