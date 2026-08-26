"""Фильтры и веса МКБ для подбора клинических протоколов под КЗ."""
from __future__ import annotations

import re
from typing import Any

_ADMIN_TITLE_MARKERS = (
    "об утверждении клинических протоколов",
    "об утверждении клинического протокола",
    "об утверждении некоторых клинических",
    "постановление министерства",
)

_SPINE_ICD_ROOTS = frozenset({"M51", "M53", "M54"})
_SPINE_NEEDLES = (
    "ишиас", "радикул", "люмбо", "позвоноч", "вертеброген", "межпозвон", "м54", "m54",
    "остеохондр", "спондил",
)
_SPINE_WRONG = (
    "эпилепс", "судорож", "вестибул", "венозн", "тромбоз", "тгв", "флеботромб",
    "тромбоэмбол", "варикоз",
)


def _icd_root(code: str) -> str:
    c = (code or "").upper().strip()
    return c[:3] if len(c) >= 3 else c


def _card_blob(card: dict[str, Any]) -> str:
    return ((card.get("title") or "") + " " + (card.get("source_path") or "")).lower()


def is_administrative_protocol(card: dict[str, Any]) -> bool:
    """Приказы об утверждении КП - не клинический эталон для сравнения."""
    blob = _card_blob(card)
    if any(m in blob for m in _ADMIN_TITLE_MARKERS):
        return True
    kind = str(card.get("protocol_kind") or "").lower()
    if kind in (
        "general_program",
        "general_care",
        "screening_dispanser",
        "admin",
        "rehab",
        "rehabilitation",
        "algorithm",
    ):
        return True
    icd = list(card.get("icd10_all") or card.get("icd10_primary") or [])
    if not icd and "утверждении" in blob and "клиническ" in blob:
        return True
    return False


def icd_fit_for_card(
    card: dict[str, Any],
    icd_codes: list[str],
) -> list[dict[str, Any]]:
    """Метки МКБ с весом подходимости для карточки протокола."""
    card_icd = [str(x).upper() for x in (card.get("icd10_primary") or []) if x]
    card_set = set(card_icd)
    card_roots = {_icd_root(c) for c in card_icd}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in icd_codes:
        code = str(raw).upper().strip()
        if not code or code in seen:
            continue
        root = _icd_root(code)
        weight = 0.0
        if code in card_set:
            weight = 1.0
        elif root in card_roots:
            weight = 0.85
        elif any(code.startswith(r) or r.startswith(root) for r in card_roots if r):
            weight = 0.55
        if weight > 0:
            seen.add(code)
            out.append({"code": code, "weight": round(weight, 2)})
    out.sort(key=lambda x: (-x["weight"], x["code"]))
    return out


def format_icd_fit_labels(fit: list[dict[str, Any]], *, limit: int = 4) -> str:
    if not fit:
        return ""
    parts = [f"{x['code']} ({x['weight']:.2f})" for x in fit[:limit]]
    return ", ".join(parts)


def clinical_relevance_multiplier(
    card: dict[str, Any],
    *,
    icd_codes: list[str] | None = None,
    complaints: list[str] | str | None = None,
    ambulatory: bool = True,
) -> float:
    """Дополнительный множитель релевантности нозологии и формата помощи."""
    blob = _card_blob(card)
    mult = 1.0
    icd_list = [str(c).upper() for c in (icd_codes or []) if c]
    icd_roots = {_icd_root(c) for c in icd_list}

    if icd_roots & _SPINE_ICD_ROOTS or any(c.startswith("M54") for c in icd_list):
        if any(n in blob for n in _SPINE_NEEDLES):
            mult *= 1.22
        if any(n in blob for n in _SPINE_WRONG):
            mult *= 0.07

    compl_blob = " ".join(complaints) if isinstance(complaints, list) else str(complaints or "")
    compl_low = compl_blob.lower()
    if compl_low and any(n in compl_low for n in ("ишиас", "онемен", "радикул", "люмбо", "позвоноч")):
        if any(n in blob for n in _SPINE_NEEDLES):
            mult *= 1.1
        if any(n in blob for n in _SPINE_WRONG):
            mult *= 0.12

    if ambulatory:
        inpatient = bool(re.search(r"стационарн|круглосуточн|госпитализац", blob))
        amb_ok = bool(re.search(r"амбулатор|поликлиник", blob))
        if inpatient and not amb_ok:
            mult *= 0.32

    return mult
