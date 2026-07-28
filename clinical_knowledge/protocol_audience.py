"""Аудитория и читаемые названия протоколов (из имени PDF и index.csv)."""
from __future__ import annotations

import re

_PED_MARKERS = (
    "д-нас",
    "дет-нас",
    "дет нас",
    "дет_нас",
    "детс",
    "дет. нас",
    "детск",
    "детей",
    "дет ",
    " дет",
    "неонат",
    "новорожд",
    "pediatr",
    "дет возраста",
)
_ADULT_MARKERS = (
    "взросл",
    "взр ",
    "взр.",
    "взр_нас",
    "в-нас",
    " в нас",
    "вз н",
    "вз_н",
)


def norm_audience_blob(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("_", " ").replace("-", " ")).strip()


def infer_protocol_audience(path: str, title: str = "") -> str | None:
    """pediatric | adult | mixed | None - по имени файла/заголовка."""
    blob = norm_audience_blob(f"{path} {title}")
    has_p = any(m in blob for m in _PED_MARKERS)
    has_a = any(m in blob for m in _ADULT_MARKERS)
    if has_p and has_a:
        return "mixed"
    if has_p:
        return "pediatric"
    if has_a:
        return "adult"
    return None


def audience_hint_ru(audience: str | None) -> str | None:
    if audience == "pediatric":
        return "детское население"
    if audience == "adult":
        return "взрослое население"
    if audience == "mixed":
        return "дети и взрослые"
    return None


_TITLE_ABBR_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bд-нас\b", re.I), "детское население"),
    (re.compile(r"\bдет-нас\b", re.I), "детское население"),
    (re.compile(r"\bдетс\s+нас\b", re.I), "детское население"),
    (re.compile(r"\bдет\s+нас\b", re.I), "детское население"),
    (re.compile(r"\bвзр\.?\s*нас(?:еление)?\b", re.I), "взрослое население"),
    (re.compile(r"\bв-нас\b", re.I), "взрослое население"),
    (re.compile(r"\bвзр\s+нас\b", re.I), "взрослое население"),
    (re.compile(r"\bпост\.?\s*мз\b", re.I), "пост. МЗ"),
    (re.compile(r"\bпостановление\s+мз\b", re.I), "постановление МЗ"),
    (re.compile(r"\bстац\.?\s+услов", re.I), "стационарных условиях"),
    (re.compile(r"\bамбул\.?\b", re.I), "амбулаторных"),
)


def expand_protocol_title_abbreviations(name: str) -> str:
    """Расшифровка типичных сокращений Минздрава в названии протокола."""
    out = str(name or "").strip()
    if not out:
        return ""
    for pat, repl in _TITLE_ABBR_REPLACEMENTS:
        out = pat.sub(repl, out)
    out = out.replace("»", "").replace("«", "")
    out = re.sub(r"\s+", " ", out).strip(" .,-")
    return out


def is_synthetic_summary_excerpt(text: str) -> bool:
    """Фрагмент из summary_to_rag, не цитата из PDF."""
    t = (text or "").strip()
    if not t:
        return False
    if t.startswith("Протокол:") and "Нозология:" in t:
        return True
    if t.startswith("Протокол:") and "МКБ-10:" in t and "Рубрика:" in t:
        return True
    return False
