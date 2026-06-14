"""Единый реестр нозологий: маркеры текста КЗ, МКБ, ключевые слова в path/title протокола."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ConditionDef:
    condition_id: str
    text_markers: tuple[str, ...]
    icd_prefixes: tuple[str, ...]
    card_keywords: tuple[str, ...]
    path_hints: tuple[str, ...] = ()


def _defs() -> list[ConditionDef]:
    g = [
        ("gerd", ("гэрб", "gerd", "рефлюкс", "изжог"), ("K21",), ("гэрб", "рефлюкс", "пищевод", "желудк", "двенадцат"), ("пищевода_желудка", "гэрб")),
        ("gastritis", ("гастрит",), ("K29",), ("гастрит",), ("гастрит",)),
        ("peptic_ulcer", ("язв", "гастродуоден"), ("K25", "K26", "K27", "K28"), ("язв", "гастродуоден"), ("язвенн", "гастродуоден")),
        ("functional_dyspepsia", ("диспепс",), ("K30",), ("диспепс",), ("диспепс",)),
        ("crohn", ("крон", "болезн крона"), ("K50",), ("крон", "k50"), ("крон", "k50")),
        ("ulcerative_colitis", ("язвенн", "колит"), ("K51",), ("колит", "k51"), ("язвенн", "колит", "k51")),
        ("celiac", ("целиак",), ("K90",), ("целиак",), ("целиак",)),
        ("acute_pancreatitis", ("панкреат",), ("K85",), ("панкреат",), ("панкреат",)),
        ("acute_appendicitis", ("аппендицит",), ("K35", "K37"), ("аппендицит",), ("аппендицит",)),
        ("acute_cholecystitis", ("холецист",), ("K81",), ("холецист",), ("холецист",)),
        ("intestinal_obstruction", ("непроходим",), ("K56",), ("непроходим",), ("непроходим",)),
        ("intussusception", ("инвагинац",), ("K56.1",), ("инвагинац",), ("инвагинац",)),
        ("incarcerated_hernia", ("грыж", "ущемл"), ("K40",), ("грыж",), ("грыж",)),
        ("foreign_body_gi", ("инородн",), ("T18",), ("инородн",), ("инородн",)),
        ("gi_bleeding", ("кровотеч",), ("K92",), ("кровотеч",), ("кровотеч",)),
        ("perforated_peptic_ulcer", ("перфора",), ("K26.1", "K25.1"), ("перфора",), ("перфоратив",)),
        ("abdominal_trauma", ("травм",), ("S36",), ("травм", "живот"), ("травм",)),
        ("acute_bronchitis", ("бронхит",), ("J20", "J21"), ("бронхит",), ("бронхит",)),
        ("pneumonia", ("пневмон",), ("J12", "J13", "J14", "J15", "J16", "J17", "J18"), ("пневмон",), ("пневмон",)),
        ("bronchial_asthma", ("астм",), ("J45", "J46"), ("астм",), ("астм",)),
        ("copd", ("хобл", "бронхит хрон"), ("J44",), ("хобл", "обструкт"), ("хобл",)),
        ("tuberculosis", ("туберкул",), ("A15", "A16", "A17", "A18", "A19"), ("туберкул",), ("туберкул",)),
        ("myocardial_infarction", ("инфаркт",), ("I21", "I22"), ("инфаркт",), ("инфаркт",)),
        ("angina_pectoris", ("стенокард",), ("I20",), ("стенокард",), ("стенокард",)),
        ("cardiac_arrhythmia", ("аритми",), ("I47", "I48", "I49"), ("аритми",), ("аритми",)),
        ("heart_failure", ("сердечн", "недостаточ"), ("I50",), ("недостаточ", "сердечн"), ("сердечн",)),
        ("hypertension", ("гипертон", "артериальн"), ("I10", "I11", "I12", "I13", "I15"), ("гипертон",), ("гипертон",)),
        ("stroke", ("инсульт", "острый наруш"), ("I60", "I61", "I62", "I63", "I64"), ("инсульт",), ("инсульт",)),
        ("epilepsy", ("эпилепс",), ("G40", "G41"), ("эпилепс",), ("эпилепс",)),
        ("multiple_sclerosis", ("рассеян", "склероз"), ("G35",), ("рассеян", "склероз"), ("рассеян",)),
        ("migraine", ("мигрен",), ("G43",), ("мигрен",), ("мигрен",)),
        ("carcinoma", ("карцином", "рак"), ("C",), ("карцином", "опухол", "рак"), ("карцином", "опухол")),
        ("neoplasm", ("опухол", "новообраз"), ("C", "D"), ("опухол", "новообраз"), ("novoobraz", "опухол")),
        ("lymphoma", ("лимфом",), ("C81", "C82", "C83", "C84", "C85"), ("лимфом",), ("лимфом",)),
        ("leukemia", ("лейкоз",), ("C91", "C92", "C93", "C94", "C95"), ("лейкоз",), ("лейкоз",)),
        ("diabetes_mellitus", ("диабет", "сахарн"), ("E10", "E11", "E13", "E14"), ("диабет", "сахарн"), ("диабет", "сахарн")),
        ("thyroid_disease", ("гипотиреоз", "гипертиреоз", "тиреоидит"), ("E03", "E04", "E05", "E06"), ("щитовид",), ("щитовид",)),
        ("obesity", ("ожирен",), ("E66",), ("ожирен",), ("ожирен",)),
        ("renal_failure", ("почечн", "недостаточ"), ("N17", "N18", "N19"), ("почечн", "недостаточ"), ("почечн",)),
        ("urolithiasis", ("мочекамен",), ("N20", "N21"), ("мочекамен",), ("мочекамен",)),
        ("prostate_disease", ("простат",), ("N40", "N41"), ("простат",), ("простат",)),
        ("arthritis", ("артрит",), ("M05", "M06", "M13"), ("артрит",), ("артрит",)),
        ("osteoarthritis", ("остеоартроз",), ("M15", "M16", "M17", "M19"), ("остеоартроз",), ("остеоартроз",)),
        ("gout", ("подагр",), ("M10",), ("подагр",), ("подагр",)),
        ("sle", ("скв", "волчанк"), ("M32", "L93"), ("скв", "волчанк"), ("скв",)),
        ("fracture", ("перелом",), ("S", "T"), ("перелом",), ("перелом",)),
        ("osteoporosis", ("остеопороз",), ("M80", "M81"), ("остеопороз",), ("остеопороз",)),
        ("psoriasis", ("псориаз",), ("L40",), ("псориаз",), ("псориаз",)),
        ("dermatitis", ("дерматит", "экзем"), ("L20", "L23", "L24", "L30"), ("дерматит",), ("дерматит",)),
        ("influenza", ("грипп",), ("J09", "J10", "J11"), ("грипп",), ("грипп",)),
        ("covid19", ("covid", "коронавирус", "covid-19"), ("U07", "J12.8"), ("covid", "коронавирус"), ("covid",)),
        ("cataract", ("катаракт",), ("H25", "H26"), ("катаракт",), ("катаракт",)),
        ("glaucoma", ("глауком",), ("H40",), ("глауком",), ("глауком",)),
        ("depression", ("депресс",), ("F32", "F33"), ("депресс",), ("депресс",)),
        ("schizophrenia", ("шизофрен",), ("F20",), ("шизофрен",), ("шизофрен",)),
        ("dental_caries", ("кариес",), ("K02",), ("кариес",), ("кариес", "stomatolog")),
        ("pulpitis", ("пульпит",), ("K04",), ("пульпит",), ("пульпит",)),
        ("anemia", ("анем",), ("D50", "D51", "D52", "D53", "D55", "D56", "D57", "D58", "D59", "D60", "D61", "D62", "D63", "D64"), ("анем",), ("анем",)),
        ("pregnancy", ("беремен", "гестаци"), ("O", "Z33", "Z34"), ("беремен",), ("беремен",)),
        (
            "deep_vein_thrombosis",
            ("тромб", "флеботромб", "тромбофлеб", "флебит", "тромбоз вен", "тгв", "тромбоэмбол"),
            ("I80", "I81", "I82", "I83", "I87"),
            ("тромб", "вен", "флеб", "тромбоз"),
            ("tromboz", "ven", "tromb"),
        ),
    ]
    return [ConditionDef(cid, tm, icd, ck, ph) for cid, tm, icd, ck, ph in g]


CONDITIONS: list[ConditionDef] = _defs()
CONDITION_BY_ID: dict[str, ConditionDef] = {c.condition_id: c for c in CONDITIONS}


def infer_conditions_hints(text_low: str, icd_codes: Iterable[str]) -> list[str]:
    """Нозологии по тексту КЗ и кодам МКБ."""
    from clinical_knowledge.rule_family_gates import is_oncology_icd

    low = text_low or ""
    icd_up = [str(c).upper().strip() for c in icd_codes if c]
    out: list[str] = []
    for c in CONDITIONS:
        if any(m in low for m in c.text_markers):
            out.append(c.condition_id)
            continue
        if c.condition_id == "neoplasm":
            if any(is_oncology_icd(code) for code in icd_up):
                out.append(c.condition_id)
            continue
        for pref in c.icd_prefixes:
            if any(code.startswith(pref.upper()) for code in icd_up):
                out.append(c.condition_id)
                break
    return list(dict.fromkeys(out))


def score_card_for_hint(hint: str, blob: str, icd_list: list[str]) -> float:
    """Дополнительный score сопоставления карточки протокола с hint."""
    c = CONDITION_BY_ID.get(hint)
    if not c:
        return 0.0
    score = 0.0
    if any(kw in blob for kw in c.card_keywords):
        score += 28.0
    for pref in c.icd_prefixes:
        if any(code.startswith(pref.upper()) for code in icd_list):
            if any(kw in blob for kw in c.card_keywords):
                score += 12.0
            break
    return score


def source_path_condition_hints() -> list[tuple[str, tuple[str, ...]]]:
    """Для rules_from_corpus: path needles по condition_id."""
    return [(c.condition_id, c.path_hints or c.card_keywords[:2]) for c in CONDITIONS if c.path_hints or c.card_keywords]
