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
            "нервной систем",
            "урolog",
            "уролог",
            "дет_",
            "д-нас",
            "детс",
            "pediatr",
            "стомат",
            "заболеваниями н",
            "neyro",
            "нейро",
        ),
        "suppress_when_active": ("heart_failure", "migraine"),
    },
    {
        "id": "gynecology",
        "query_markers": ("менстру", "маточ", "яичник", "эндометр", "миома", "кольпит", "цервик", "маstит", "лактац", "мастит"),
        "icd_prefixes": ("N80", "N81", "N82", "N83", "N84", "N85", "N86", "N87", "N88", "N89", "N90", "N91", "N92", "N93", "N94", "N95"),
        "slugs": ("akusherstvo-ginekologiya",),
        "title_strong": ("ginekolog", "гинекolog", "акушer", "акушer", "маточ", "менстру", "эндометр", "яичник", "mastit", "маstit", "лактац", "мастит"),
        "title_weak": ("женск", "репродукт"),
        "title_wrong": ("эндокрин", "гипофиз", "надпочеч", "cardio"),
    },
    {
        "id": "orvi_uri",
        "query_markers": ("орви", "орз", "простуд", "респираторн", "насморк"),
        "icd_prefixes": ("J06", "J00", "J11", "J20", "J21"),
        "slugs": (
            "infektsionnye-zabolevaniya",
            "pulmonologiya-ftiziatriya",
            "otorinolaringologiya",
            "terapiya",
            "pediatriya",
        ),
        "title_strong": ("орви", "орз", "респиратор", "простуд", "гриpp", "грипп"),
        "title_weak": ("инфекц", "respir"),
        "title_wrong": (
            "гепатит",
            "hepat",
            "риносинус",
            "sinusit",
            "синусит",
            "трансплант",
            "паллиат",
            "онкolog",
            "оториноларингологическ",
            "заболеваниями в-нас",
            "заболеваниями в нас",
            "дистресс",
        ),
    },
    {
        "id": "otitis",
        "query_markers": ("отит", "боль в ух", "ухо ", " ухе", "уха "),
        "icd_prefixes": ("H65", "H66", "H67"),
        "slugs": ("otorinolaringologiya",),
        "title_strong": ("отит", "оторin", "уха", "средний отит"),
        "title_weak": ("лор", "оториноларинг"),
        "title_wrong": ("буллез", "дерматит", "невроген", "урolog", "нейрохирург", "нервной систем"),
    },
    {
        "id": "hypertension",
        "query_markers": ("гипертон", "гипертенз", "давлен", "артериальн", "i10"),
        "icd_prefixes": ("I10", "I11", "I12", "I13", "I15"),
        "slugs": ("bolezni-sistemy-krovoobrashcheniya", "terapiya"),
        "title_strong": ("гипертенз", "артериальн", "гипертон", "давлен"),
        "title_weak": ("сердечн", "krovoobrash", "кровообращ", "cardio", "кардиол"),
        "title_wrong": (
            "вен ",
            "варикоз",
            "тромбофлеб",
            "мочев",
            "уролог",
            "невроген",
            "д-нас",
            "дет нас",
            "дет_нас",
            "детс",
            "pediatr",
        ),
    },
    {
        "id": "coronary",
        "query_markers": ("ишем", "стенокард", "коронар", "инфаркт миокард"),
        "icd_prefixes": ("I20", "I21", "I22", "I23", "I24", "I25"),
        "slugs": ("bolezni-sistemy-krovoobrashcheniya",),
        "title_strong": ("ишем", "стенокард", "инфаркт", "коронар", "ишемическ"),
        "title_weak": ("krovoobrash", "кровообращ", "cardio", "кардиол"),
        "title_wrong": (
            "оторин",
            "оторin",
            "лор",
            "ларинг",
            "фаринг",
            "тромбоз",
            "глубоких вен",
            "периферич",
            "д-нас",
            "дет нас",
        ),
    },
    {
        "id": "heart_failure",
        "query_markers": ("сердечн", "недостат", "одыш", "отек", "отёк"),
        "icd_prefixes": ("I50", "I11", "I13"),
        "slugs": ("bolezni-sistemy-krovoobrashcheniya",),
        "title_strong": ("сердечн", "недостат", "cardio", "сердечной недостат"),
        "title_weak": ("krovoobrash", "кровообращ"),
        "title_wrong": (
            "эндокрин",
            "почек",
            "nefrolog",
            "тромбоз",
            "глубоких вен",
            "периферич",
            "варикоз",
        ),
    },
    {
        "id": "sinusitis",
        "query_markers": ("риносинус", "sinusit", "синусит", "синус"),
        "icd_prefixes": ("J32", "J01"),
        "slugs": ("otorinolaringologiya",),
        "title_strong": ("риносинус", "sinusit", "синусит", "синус"),
        "title_weak": ("otorin", "лор"),
        "title_wrong": ("vich", "вич", "psikhiatr", "психическ", "инфекциями кожи"),
    },
    {
        "id": "stroke",
        "query_markers": ("инсульт", "cerebro", "ишемическ инс", "геморrag"),
        "icd_prefixes": ("I60", "I61", "I62", "I63", "I64", "G45"),
        "slugs": ("nevrologiya-neyrokhirurgiya",),
        "title_strong": ("инсульт", "nevrolog", "нейро", "cerebro", "сосудист", "нейроишем", "реабилита"),
        "title_weak": ("krovoobrash",),
        "title_wrong": (
            "инфаркт миокард",
            "стенокард",
            "коронар",
            "акушер",
            "беремен",
            "женщинам",
        ),
    },
    {
        "id": "asthma_pulmonology",
        "query_markers": ("астм", "бронхosp", "бронхосп", "wheez", "сibil"),
        "icd_prefixes": ("J45", "J46"),
        "slugs": ("pulmonologiya-ftiziatriya",),
        "title_strong": ("астм", "pulmonolog", "бронхosp", "бронхосп"),
        "title_weak": ("бронхит", "respir", "респиратор"),
        "title_wrong": (
            "оториноларингологическ",
            "заболеваниями в-нас",
            "заболеваниями в нас",
            "лор",
            "фаринг",
            "риносинус",
        ),
    },
    {
        "id": "allergy_rhinitis",
        "query_markers": ("аллерг", "ринит", "чихан", "зуд в нос", "поллиноз"),
        "icd_prefixes": ("J30",),
        "slugs": ("allergologiya-immunologiya", "otorinolaringologiya"),
        "title_strong": ("аллерг", "ринит", "allerg"),
        "title_weak": ("otorin", "лор"),
        "title_wrong": ("буллез", "дерматит", "экзем", "придатков кожи", "крапивниц", "оторinоларингologическ"),
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
        "title_strong": ("ожог", "термическ", "обожжен", "термической травм"),
        "title_weak": ("реаним", "anestez", "khirurg"),
        "title_wrong": (
            "урolog",
            "уролог",
            "мочев",
            "гинекolog",
            "травмой живота",
            "травма живота",
            "травм живота",
            "живота в стацион",
            "скорой неотложной",
            "неотложной медицинской помощи",
            "респираторного дистресс",
        ),
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
        "title_strong": ("цистит", "urolog", "урolog", "urologiya", "мочев", "простат"),
        "title_weak": ("nefrolog", "почек"),
        "title_wrong": ("ожог", "cardio", "nefrologiya", "желчн", "gastroenterolog"),
    },
    {
        "id": "nephrology",
        "query_markers": ("пиелонеф", "глomerul", "гломерул", "нефроп", "почечн недостат"),
        "icd_prefixes": ("N10", "N11", "N12", "N13", "N15", "N18", "N19"),
        "slugs": ("nefrologiya",),
        "title_strong": ("пиелонеф", "nefrolog", "нефrolog", "нефрolog", "почек", "почеч"),
        "title_weak": ("urolog", "урolog"),
        "title_wrong": ("urologiya", "урolog", "цистит", "простат"),
    },
    {
        "id": "infectious",
        "query_markers": ("диар", "инфекц", "лихорад", "vich", "вич", "сальmonell", "сальмонел"),
        "icd_prefixes": ("A00", "A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "B20", "B21"),
        "slugs": ("infektsionnye-zabolevaniya", "gastroenterologiya"),
        "title_strong": ("infektsion", "инфекц", "диар", "кишеч", "vich", "вич"),
        "title_weak": ("gastro", "гастр"),
        "title_wrong": ("перфорат", "кровотеч", "паллиат", "онкolog"),
    },
    {
        "id": "wound",
        "query_markers": ("рана ", " рана", "раны ", "раной", "ранен", "порез", "колот"),
        "icd_prefixes": ("S01", "S11", "S21", "S31", "S41", "S51", "S61", "S71", "S81", "S91", "T01", "T11", "T14"),
        "slugs": ("khirurgiya", "travmatologiya-ortopediya", "anesteziologiya-reanimatologiya"),
        "title_strong": ("ран", "ранен", "раной", "khirurg", "хирург", "travm"),
        "title_weak": ("неотлож", "скорой"),
        "title_wrong": (
            "травмой живота",
            "травма живота",
            "травм живота",
            "огнестрел",
            "огнестр",
            "перелом",
        ),
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
        "title_wrong": ("орви", "гастрит", "паллиат", "фармакотерап", "симптомов при", "глаза", "oftalm", "офтальм"),
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
    {
        "id": "gastroenterology",
        "query_markers": (
            "запор",
            "вздут",
            "метеор",
            "дисхез",
            "дефекац",
            "изжог",
            "тошн",
            "рвот",
            "гастр",
            "кишеч",
            "срк",
            "диспепс",
            "стул",
            "диар",
            "гэрб",
            "рефлюкс",
            "кишечник",
        ),
        "icd_prefixes": ("K21", "K25", "K26", "K29", "K50", "K51", "K58", "K59", "K80", "K81", "K85", "K92"),
        "slugs": ("gastroenterologiya",),
        "title_strong": (
            "gastro",
            "гастр",
            "кишеч",
            "кишечник",
            "запор",
            "дефекац",
            "пищевар",
            "желуд",
            "желч",
            "печен",
            "поджелуд",
            "колит",
            "гастрит",
            "эзофаг",
            "моторно",
            "эвакуатор",
        ),
        "title_weak": ("абdom", "брюш"),
        "title_wrong": (
            "травм",
            "огнестрел",
            "огнестр",
            "ранени",
            "ранен",
            "ранами",
            "пулев",
            "ножев",
            "проникающ",
        ),
    },
]


def _query_blob(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").lower()).strip()


def _title_blob(path: str, title: str) -> str:
    """Нормализация path/title для подстрок (PDF часто с подчёркиваниями)."""
    raw = f"{path} {title}".lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", raw).strip()


def _marker_in_blob(marker: str, blob: str) -> bool:
    m = re.sub(r"\s+", " ", str(marker or "").lower().replace("_", " ").replace("-", " ")).strip()
    return bool(m) and m in blob


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


def _route_by_id(rid: str) -> dict[str, Any] | None:
    return next((r for r in _CLINICAL_ROUTES if r["id"] == rid), None)


def detect_clinical_route_ids(query: str, icd_codes: list[str] | None = None) -> list[str]:
    """Активные клинические маршруты по тексту и МКБ (порядок приоритета)."""
    ql = _query_blob(query)
    codes = list(icd_codes or [])
    out: list[str] = []
    icd_only: set[str] = set()
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
            icd_only.add(rid)

    # Беременность + «головная боль/отёки» не должны тянуть мигрень/СН; только акушерство.
    if "pregnancy" in out and any(m in ql for m in ("беремен", "роды", "плацент", "гест")):
        preg = _route_by_id("pregnancy")
        suppress = set(preg.get("suppress_when_active") or ()) if preg else set()
        filtered: list[str] = []
        for rid in out:
            if rid in suppress and rid in icd_only:
                continue
            if rid in suppress and rid not in icd_only:
                continue
            filtered.append(rid)
        if "pregnancy" not in filtered:
            filtered.insert(0, "pregnancy")
        out = filtered
    preg_ctx = any(
        m in ql for m in ("беремен", "роды", "плацент", "гестоз", "преэклам", "эклам", "послерод")
    ) or "контекст подбора: беремен" in ql
    if "pregnancy" in out and "pregnancy" in icd_only and not preg_ctx:
        out = [r for r in out if r != "pregnancy"]
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
    blob = _title_blob(path, title)
    delta = 0.0
    matched: list[str] = []
    for rid in route_ids:
        route = next((r for r in _CLINICAL_ROUTES if r["id"] == rid), None)
        if not route:
            continue
        strong = sum(1 for m in route.get("title_strong") or () if _marker_in_blob(m, blob))
        weak = sum(1 for m in route.get("title_weak") or () if _marker_in_blob(m, blob))
        wrong = sum(1 for m in route.get("title_wrong") or () if _marker_in_blob(m, blob))
        touched = False
        if wrong:
            pen = 14.0 + 3.0 * wrong
            if rid == "pregnancy":
                pen += 6.0
            delta -= pen
            touched = True
        if strong:
            boost = 8.0 + 3.0 * strong + 1.0 * weak
            if rid == "pregnancy":
                boost += 4.0
            delta += boost
            touched = True
        elif weak and not wrong:
            delta += 2.0 * weak
            touched = True
        if touched:
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
    blob = _title_blob(path, title)
    best: float | None = None
    for rid in route_ids:
        route = next((r for r in _CLINICAL_ROUTES if r["id"] == rid), None)
        if not route:
            continue
        if any(_marker_in_blob(m, blob) for m in route.get("title_wrong") or ()):
            if not any(_marker_in_blob(m, blob) for m in route.get("title_strong") or ()):
                continue
        strong = sum(1 for m in route.get("title_strong") or () if _marker_in_blob(m, blob))
        weak = sum(1 for m in route.get("title_weak") or () if _marker_in_blob(m, blob))
        if strong == 0 and weak == 0:
            continue
        score = base + 10.0 + 6.0 * strong + 2.0 * weak
        if best is None or score > best:
            best = score
    return best
