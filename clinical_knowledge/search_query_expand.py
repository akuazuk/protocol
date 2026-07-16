"""Детерминированное расширение клинического запроса и уточняющие оси (MSK / хромота).

Без LLM: добавляет термины корпуса и коды-якоря, чтобы RAG не уезжал в
«бедренный нерв / вену / грыжу» и R95 от слова «ребенка».
"""
from __future__ import annotations

import re
from typing import Any

_PED_MARKERS = (
    r"\bдет",
    r"\bдети",
    r"\bребен",
    r"\bребён",
    r"\bноворожд",
    r"\bподрост",
    r"\bшкольник",
    r"\bлет\b",
    r"контекст подбора:\s*дет",
)
_HIP_MARKERS = (
    "бедр",
    "тазобедрен",
    "тбс",
    "хромот",
    "прихрам",
    "походк",
    "кокс",
    "вертлюж",
)
_LIMP_MARKERS = ("хромот", "прихрам", "нарушен", "походк")
_FEVER_MARKERS = ("температ", "лихорад", "жар", "озноб", "субфебрил")
_TRAUMA_MARKERS = ("травм", "ушиб", "удар", "паден", "перелом", "вывих")
_NIGHT_MARKERS = ("ночн", "ночью", "в покое", "поко")


def _norm(q: str) -> str:
    return (q or "").lower().replace("ё", "е")


def is_pediatric_context(query: str) -> bool:
    q = _norm(query)
    if not q:
        return False
    if re.search(r"\b([1-9]|1[0-7])\s*лет", q):
        return True
    return any(re.search(p, q) for p in _PED_MARKERS)


def is_hip_or_limp_complaint(query: str) -> bool:
    q = _norm(query)
    return any(m in q for m in _HIP_MARKERS)


def is_pediatric_hip_limp_complaint(query: str) -> bool:
    return is_pediatric_context(query) and is_hip_or_limp_complaint(query)


def needs_msk_clinical_clarify(query: str, context: dict[str, Any] | None = None) -> bool:
    """True, если стоит спросить про лихорадку / травму / ночную боль / локализацию."""
    ctx = context or {}
    if ctx.get("clarify_confirmed") or ctx.get("clarify_skipped"):
        return False
    if not is_hip_or_limp_complaint(query):
        return False
    q = _norm(query)
    has_fever = any(m in q for m in _FEVER_MARKERS)
    has_trauma = any(m in q for m in _TRAUMA_MARKERS)
    has_night = any(m in q for m in _NIGHT_MARKERS)
    has_joint = "тазобедрен" in q or "сустав" in q or "тбс" in q
    # Уже достаточно клинических якорей - шаг можно пропустить.
    if has_fever and has_trauma and (has_night or has_joint):
        return False
    return True


def msk_clarify_choices(query: str) -> list[dict[str, Any]]:
    """Кнопки уточнения для UI (multi-select)."""
    ped = is_pediatric_context(query)
    base: list[dict[str, Any]] = [
        {
            "id": "no_fever",
            "label": "Без температуры",
            "score": 90,
            "append": "без лихорадки",
        },
        {
            "id": "fever",
            "label": "Есть температура / лихорадка",
            "score": 88,
            "append": "лихорадка температура",
        },
        {
            "id": "no_trauma",
            "label": "Без травмы",
            "score": 87,
            "append": "без травмы",
        },
        {
            "id": "trauma",
            "label": "Была травма / падение",
            "score": 86,
            "append": "травма падение",
        },
        {
            "id": "night_pain",
            "label": "Боль ночью или в покое",
            "score": 84,
            "append": "боль ночью в покое",
        },
        {
            "id": "hip_joint",
            "label": "Тазобедренный сустав (ТБС)",
            "score": 92 if ped else 85,
            "append": "тазобедренный сустав ТБС",
        },
        {
            "id": "soft_tissue",
            "label": "Мягкие ткани бедра / мышцы",
            "score": 80,
            "append": "мягкие ткани бедра мышцы",
        },
    ]
    if ped:
        base.insert(
            0,
            {
                "id": "limp_gait",
                "label": "Хромота / нарушение походки",
                "score": 94,
                "append": "хромота нарушение походки",
            },
        )
    return base


def expand_clinical_query_terms(query: str) -> tuple[str, dict[str, Any]]:
    """Добавляет клинические термины для retrieve; исходный текст сохраняется в начале."""
    q = (query or "").strip()
    if len(q) < 4:
        return q, {"applied": False}
    ql = _norm(q)
    extras: list[str] = []
    meta: dict[str, Any] = {"applied": False, "profiles": []}

    if is_pediatric_hip_limp_complaint(q):
        meta["profiles"].append("pediatric_hip_limp")
        for term in (
            "тазобедренный сустав",
            "ТБС",
            "хромота",
            "нарушение походки",
            "болезнь Пертеса",
            "остеохондропатия",
            "коксит",
            "транзиторный синовит",
            "ювенильный артрит",
            "ортопедия детская",
        ):
            if term.lower().replace("ё", "е") not in ql:
                extras.append(term)

    if extras:
        expanded = q + "\n" + " ".join(extras)
        meta["applied"] = True
        meta["extra_terms"] = extras
        return expanded, meta
    return q, meta


def append_clarify_to_query(query: str, selected: list[dict[str, Any]] | None) -> str:
    """Дописывает выбранные уточнения в текст запроса (для lexicon + retrieve)."""
    q = (query or "").strip()
    if not selected:
        return q
    parts: list[str] = []
    seen: set[str] = set()
    for row in selected:
        ap = str((row or {}).get("append") or "").strip()
        if not ap or ap in seen:
            continue
        seen.add(ap)
        parts.append(ap)
    if not parts:
        return q
    block = " - Ответы на уточняющие вопросы: " + "; ".join(parts)
    # clinical_query_for_rag отрезает этот блок для lexicon - поэтому дублируем якоря и в тело.
    anchors = " ".join(parts)
    if anchors.lower() in q.lower():
        return q + block
    return q + "\n" + anchors + "\n" + block
