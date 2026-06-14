"""Маршрутизация поиска протоколов по клиническому тексту (специальность, симптомы).

Используется search_retrieval и rerank в rag_server по паттернам AI-review / probe-batch.
"""
from __future__ import annotations

import re
from typing import Any

from clinical_knowledge.diagnosis_icd import normalize_code

# id → slugs, маркеры запроса, сильные/слабые заголовки PDF, чужие домены
_CLINICAL_ROUTES: list[dict[str, Any]] = [
    {
        "id": "pregnancy",
        "query_markers": ("беремен", "роды", "плацент", "преэклам", "эклам", "гестоз", "послерод"),
        "icd_prefixes": ("O",),
        "slugs": ("akusherstvo-ginekologiya",),
        "title_strong": ("акушер", "беремен", "гинекolog", "гинеколог", "родов", "плацент", "послерод"),
        "title_weak": ("женск", "репродукт"),
        "title_wrong": (
            "невrolog",
            "неврolog",
            "нervn",
            "нервн",
            "урolog",
            "уролог",
            "дет_",
            "д-нас",
            "pediatr",
            "стомат",
            "заболеваниями н",
        ),
    },
    {
        "id": "otitis",
        "query_markers": ("отит", "боль в ух", "ухо ", " ухе", "уха "),
        "icd_prefixes": ("H65", "H66", "H67"),
        "slugs": ("otorinolaringologiya",),
        "title_strong": ("отит", "оторin", "уха", "средний отит"),
        "title_weak": ("лор", "оториноларинг"),
        "title_wrong": ("буллез", "дерматит", "невроген", "урolog"),
    },
    {
        "id": "hypertension",
        "query_markers": ("гипертон", "гипертенз", "давлен", "артериальн", "i10"),
        "icd_prefixes": ("I10", "I11", "I12", "I13", "I15"),
        "slugs": ("bolezni-sistemy-krovoobrashcheniya", "terapiya"),
        "title_strong": ("гипертенз", "артериальн", "гипертон", "кardiol", "кардиол"),
        "title_weak": ("сердечн", "krovoobrash"),
        "title_wrong": ("вен ", "варикоз", "тромбофлеб", "мочев", "уролог", "невроген"),
    },
    {
        "id": "heart_failure",
        "query_markers": ("сердечн", "недостат", "одыш", "отек", "отёк"),
        "icd_prefixes": ("I50", "I11", "I13"),
        "slugs": ("bolezni-sistemy-krovoobrashcheniya",),
        "title_strong": ("сердечн", "недостат", "cardio"),
        "title_weak": ("krovoobrash",),
        "title_wrong": ("эндокрин", "почек", "nefrolog"),
    },
    {
        "id": "allergy_rhinitis",
        "query_markers": ("аллерг", "ринит", "чихан", "зуд в нос", "поллиноз"),
        "icd_prefixes": ("J30",),
        "slugs": ("allergologiya-immunologiya", "otorinolaringologiya"),
        "title_strong": ("аллерг", "ринит", "allerg"),
        "title_weak": ("otorin", "лор"),
        "title_wrong": ("буллез", "дерматит", "экзем", "придатков кожи", "крапивниц"),
    },
    {
        "id": "lupus",
        "query_markers": ("волчан", "sle", "ревмат"),
        "icd_prefixes": ("M32", "M05", "M06", "L93"),
        "slugs": ("revmatologiya", "allergologiya-immunologiya"),
        "title_strong": ("волчан", "revmat", "ревмат", "системн"),
        "title_weak": ("иммуновоспал",),
        "title_wrong": ("буллез", "трансплант", "дерматит", "экзем"),
    },
    {
        "id": "burn",
        "query_markers": ("ожог", "термическ", "обвар", "отморож"),
        "icd_prefixes": ("T20", "T21", "T22", "T23", "T24", "T25", "T29", "T30", "T31", "T32"),
        "slugs": ("khirurgiya", "anesteziologiya-reanimatologiya"),
        "title_strong": ("ожог", "термическ", "травм", "обожжен"),
        "title_weak": ("реаним", "anestez"),
        "title_wrong": ("урolog", "уролог", "мочев", "гинекolog"),
    },
    {
        "id": "migraine",
        "query_markers": ("мигрен", "головн", "цефалг"),
        "icd_prefixes": ("G43", "G44", "R51"),
        "slugs": ("nevrologiya-neyrokhirurgiya",),
        "title_strong": ("мигрен", "головн", "nevrolog", "неврolog"),
        "title_weak": ("неврolog",),
        "title_wrong": ("неврогенн", "мочев", "bulb"),
    },
    {
        "id": "diabetes",
        "query_markers": ("диабет", "гиперглик", "полиур", "жажд"),
        "icd_prefixes": ("E10", "E11", "E13", "E14"),
        "slugs": ("endokrinologiya-narusheniya-obmena-veshchestv",),
        "title_strong": ("диабет", "endokrin", "эндокрин", "сахарн"),
        "title_weak": (),
        "title_wrong": ("cardio", "krovoobrash"),
    },
    {
        "id": "urology",
        "query_markers": ("цистит", "дизури", "мочеисп", "простат", "почечн колик"),
        "icd_prefixes": ("N30", "N39", "N40", "N20", "N21"),
        "slugs": ("urologiya", "nefrologiya"),
        "title_strong": ("цистит", "urolog", "урolog", "мочев", "простат"),
        "title_weak": ("nefrolog", "почек"),
        "title_wrong": ("ожог", "cardio"),
    },
    {
        "id": "dermatology",
        "query_markers": ("экзем", "дерматит", "псориаз", "акне", "зуд кож"),
        "icd_prefixes": ("L20", "L21", "L40", "L70"),
        "slugs": ("dermatovenerologiya",),
        "title_strong": ("дермат", "экзем", "псориаз", "кож"),
        "title_weak": (),
        "title_wrong": ("ринит", "volchan"),
    },
    {
        "id": "psychiatry",
        "query_markers": ("депресс", "тревог", "паническ", "биполяр", "шизофрен", "психоз"),
        "icd_prefixes": ("F32", "F33", "F41", "F20", "F31"),
        "slugs": ("psikhiatriya-narkologiya",),
        "title_strong": ("psikhiatr", "психиатр", "депресс", "аффектив", "психическ"),
        "title_weak": ("narkolog",),
        "title_wrong": ("орви", "бронхит"),
    },
    {
        "id": "oncology",
        "query_markers": ("рак ", "злокач", "кarcinom", "кarcin", "зно ", "онкolog", "онколог", "метастаз"),
        "icd_prefixes": ("C", "D37", "D38", "D39", "D40", "D41", "D42", "D43", "D44", "D45", "D46", "D47", "D48"),
        "slugs": ("novoobrazovaniya",),
        "title_strong": ("онкolog", "онколог", "злокач", "новообраз", "зно"),
        "title_weak": (),
        "title_wrong": ("орви", "гастрит"),
    },
    {
        "id": "ophthalmology",
        "query_markers": ("конъюнктив", "кatar", "кatar", "глаук", "зрен", "офтальм"),
        "icd_prefixes": ("H10", "H25", "H40", "H52"),
        "slugs": ("oftalmologiya",),
        "title_strong": ("oftalm", "офтальм", "глаз", "конъюнктив", "глаук"),
        "title_weak": (),
        "title_wrong": ("урolog",),
    },
    {
        "id": "stomatology",
        "query_markers": ("кaries", "кариес", "пulpit", "пульпит", "зуб", "челюст", "стомат"),
        "icd_prefixes": ("K02", "K04", "K05", "K08"),
        "slugs": ("stomatologiya",),
        "title_strong": ("stomat", "стомат", "зуб", "челюст"),
        "title_weak": (),
        "title_wrong": ("cardio",),
    },
]


def _query_blob(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").lower()).strip()


def _icd_matches_prefixes(codes: list[str], prefixes: tuple[str, ...]) -> bool:
    for raw in codes:
        c = normalize_code(str(raw))
        if not c:
            continue
        for p in prefixes:
            p = p.upper()
            if len(p) == 1:
                if c.startswith(p):
                    return True
            elif c.startswith(p):
                return True
    return False


def detect_clinical_route_ids(query: str, icd_codes: list[str] | None = None) -> list[str]:
    """Активные клинические маршруты по тексту и МКБ (порядок приоритета)."""
    ql = _query_blob(query)
    codes = list(icd_codes or [])
    out: list[str] = []
    for route in _CLINICAL_ROUTES:
        rid = str(route["id"])
        if rid in out:
            continue
        markers = route.get("query_markers") or ()
        if any(m in ql for m in markers):
            out.append(rid)
            continue
        if _icd_matches_prefixes(codes, tuple(route.get("icd_prefixes") or ())):
            out.append(rid)
    return out


def expand_slugs_for_clinical_routes(
    slugs: set[str] | list[str] | None,
    query: str,
    icd_codes: list[str] | None = None,
) -> set[str]:
    out = {s.strip() for s in (slugs or []) if s and str(s).strip()}
    for rid in detect_clinical_route_ids(query, icd_codes):
        for route in _CLINICAL_ROUTES:
            if route["id"] == rid:
                out.update(route.get("slugs") or ())
                break
    return out


def score_path_for_clinical_routes(
    path: str,
    title: str,
    *,
    route_ids: list[str],
) -> tuple[float, list[str]]:
    """Boost (>0) / penalty (<0) для rerank; возвращает (delta, matched_route_ids)."""
    blob = f"{path} {title}".lower()
    delta = 0.0
    matched: list[str] = []
    for rid in route_ids:
        route = next((r for r in _CLINICAL_ROUTES if r["id"] == rid), None)
        if not route:
            continue
        strong = sum(1 for m in route.get("title_strong") or () if m in blob)
        weak = sum(1 for m in route.get("title_weak") or () if m in blob)
        wrong = sum(1 for m in route.get("title_wrong") or () if m in blob)
        if wrong and not strong:
            delta -= 10.0 + 2.0 * wrong
            matched.append(rid)
        elif strong:
            delta += 8.0 + 3.0 * strong + 1.0 * weak
            matched.append(rid)
        elif weak:
            delta += 2.0 * weak
            matched.append(rid)
    return delta, matched


def title_match_score_for_routes(
    path: str,
    title: str,
    *,
    route_ids: list[str],
    base: float = 8.0,
) -> float | None:
    """Скор для strict title-match (None = не подходит)."""
    blob = f"{path} {title}".lower()
    best: float | None = None
    for rid in route_ids:
        route = next((r for r in _CLINICAL_ROUTES if r["id"] == rid), None)
        if not route:
            continue
        if any(m in blob for m in route.get("title_wrong") or ()):
            if not any(m in blob for m in route.get("title_strong") or ()):
                continue
        strong = sum(1 for m in route.get("title_strong") or () if m in blob)
        weak = sum(1 for m in route.get("title_weak") or () if m in blob)
        if strong == 0 and weak == 0:
            continue
        score = base + 10.0 + 6.0 * strong + 2.0 * weak
        if best is None or score > best:
            best = score
    return best
