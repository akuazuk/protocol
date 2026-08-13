"""Заголовок, пост, аудитория и kind из текста PDF / имени файла."""
from __future__ import annotations

import re
from typing import Any

from .diff import parse_post_from_name

_TITLE_QUOTED = re.compile(
    r"клинический\s+протокол\s+[«\"„]([^»\"“]{8,220})[»\"“]",
    re.I,
)
_TITLE_LINE = re.compile(
    r"клинический\s+протокол\s+[«\"]?(.{8,220}?)(?:»|\"|$)",
    re.I,
)

_REHAB = ("реабилитац", "абилитац")
_ALGO = ("алгоритм", "алгоритмы диагностики")
_ADMIN = (
    "об утверждении клинических протоколов",
    "об утверждении клинического протокола",
    "порядок оказания медицинской",
    "организация оказания медицинской",
)

_CHILD = (
    "детское население",
    "д-нас",
    "дет-нас",
    "детей",
    "новорожд",
    "неонатолог",
    "до 18 лет",
    "в возрасте до 18",
)
_ADULT = (
    "взрослое население",
    "в-нас",
    "взр. нас",
    "взрослых",
    "взросл",
)


def _blob(*parts: str) -> str:
    return " ".join(p for p in parts if p).lower().replace("ё", "е")


def extract_title(text: str, filename: str = "") -> str:
    head = (text or "")[:8000]
    m = _TITLE_QUOTED.search(head)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip(" .;")
    m = _TITLE_LINE.search(head)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip(" .;")
    stem = re.sub(r"\.(pdf|doc|docx)$", "", filename or "", flags=re.I)
    stem = re.sub(r"^кп[_\s]*", "", stem, flags=re.I)
    return re.sub(r"[_]+", " ", stem).strip()


def classify_protocol_kind(title: str, text: str = "") -> str:
    b = _blob(title, (text or "")[:4000])
    if any(x in b for x in _ADMIN):
        return "admin"
    if any(x in b for x in _REHAB):
        return "rehab"
    if any(x in b for x in _ALGO):
        return "algorithm"
    return "clinical"


def infer_audience(title: str, filename: str = "", text: str = "") -> str:
    b = _blob(title, filename, (text or "")[:6000])
    child = any(x in b for x in _CHILD)
    adult = any(x in b for x in _ADULT)
    if child and not adult:
        return "child"
    if adult and not child:
        return "adult"
    if child and adult:
        return "any"
    return "any"


def extract_protocol_metadata(
    *,
    text: str = "",
    filename: str = "",
    source_path: str = "",
) -> dict[str, Any]:
    title = extract_title(text, filename)
    post = parse_post_from_name(filename) or parse_post_from_name(text[:12000] if text else "")
    kind = classify_protocol_kind(title, text)
    audience = infer_audience(title, filename or source_path, text)
    approval_date, approval_number = (post or (None, None))
    return {
        "title": title,
        "protocol_kind": kind,
        "audience": audience,
        "approval_date": approval_date,
        "approval_number": approval_number,
        "clinical_for_score": kind == "clinical",
    }
