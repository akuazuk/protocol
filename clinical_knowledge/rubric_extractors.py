"""Рубрико-специфичные извлечения из текста КЗ (ТЗ раздел 22).

Аддитивный модуль: не меняет существующий парсинг, а дополняет результат
анализа набором профильных терминов и числовых измерений, релевантных
конкретной рубрике (специальности) протокола.

Две части:
  1. RUBRIC_TERMS — словарь «slug рубрики -> ключевые понятия» строго по
     разделу 22 ТЗ. Детектируется простое вхождение подстроки (lower).
  2. MEASUREMENT_PATTERNS — регэкспы для числовых клинических величин
     (TNM, стадия, HbA1c, СКФ, спирометрия, сатурация, СРБ, СОЭ, DAS28,
     ВГД, ПСА, фракция выброса, срок беременности, масса тела и т.д.).

Всё обёрнуто в try/except, чтобы любой сбой не ломал основной пайплайн.
"""
from __future__ import annotations

import re
from typing import Any

# --- 22.1 .. 22.24: ключевые понятия по рубрикам (slug каталога) ---
RUBRIC_TERMS: dict[str, list[str]] = {
    "akusherstvo-ginekologiya": [
        "беременность", "срок беременности", "роды", "послеродов", "плод",
        "госпитализаци", "оперативн", "маршрутизаци", "гестаци",
    ],
    "allergologiya-immunologiya": [
        "аллерген", "реакц", "анафилакс", "иммунодефицит", "кожные пробы",
        "ige", "противопоказан", "неотложн",
    ],
    "anesteziologiya-reanimatologiya": [
        "интенсивн", "критическ", "шкал", "мониторинг", "вентиляц",
        "инфузионн", "неотложн",
    ],
    "bolezni-sistemy-krovoobrashcheniya": [
        "артериальн", "давлени", "сердечн", "недостаточност", "ибс", "инфаркт",
        "экг", "эхокг", "липидн", "антикоагулянт", "антиагрегант", "госпитализаци",
    ],
    "gastroenterologiya": [
        "жкт", "эндоскоп", "h. pylori", "хеликобактер", "пищевод", "желуд",
        "кишечник", "печен", "желчн", "поджелудочн", "диет", "длительность",
    ],
    "gematologiya": [
        "анеми", "гемобластоз", "коагулопат", "общий анализ крови", "ферритин",
        "коагулограмм", "трансфуз", "госпитализаци",
    ],
    "dermatovenerologiya": [
        "кожн", "высыпан", "локализац", "иппп", "лабораторн", "наружн",
        "системн", "фотозащит", "противопоказан",
    ],
    "zabolevaniya-perinatalnogo-perioda": [
        "новорожд", "гестаци", "масса тела", "врожденн", "перинатальн",
        "интенсивн", "риск",
    ],
    "infektsionnye-zabolevaniya": [
        "возбудител", "путь передачи", "эпидемиолог", "пцр", "серолог",
        "антибактериальн", "противовирусн", "изоляц", "профилактик",
    ],
    "nevrologiya-neyrokhirurgiya": [
        "неврологическ", "дефицит", "шкал", "нейровизуализац", "ээг",
        "инсульт", "эпилепс", "нейрохирург", "реабилитац",
    ],
    "nefrologiya": [
        "скф", "креатинин", "протеинур", "альбуминур", "мочевой синдром",
        "хбп", "диализ", "нефропротектив",
    ],
    "novoobrazovaniya": [
        "локализац", "стади", "tnm", "морфолог", "гистолог", "маркер",
        "химиотерап", "лучев", "хирургическ", "диспансерн",
    ],
    "otorinolaringologiya": [
        "лор", "слух", "носов", "эндоскоп", "аудиометр", "операц",
    ],
    "oftalmologiya": [
        "остров зрения", "острота зрения", "внутриглазн", "глазн дно",
        "поля зрения", "офтальмоскоп", "oct", "глауком", "катаракт", "неотложн",
    ],
    "palliativnaya-pomoshch": [
        "боль", "шкал", "функциональн", "уход", "маршрутизац", "обезболиван",
        "поддерживающ",
    ],
    "psikhiatriya-narkologiya": [
        "психическ", "суицид", "зависимост", "интоксикац", "абстиненц",
        "госпитализац", "психофармак", "противопоказан",
    ],
    "pulmonologiya-ftiziatriya": [
        "кашель", "одышк", "спирометр", "сатураци", "рентген", "кт",
        "туберкул", "мокрот", "антибактериальн", "противотуберкулезн",
    ],
    "revmatologiya": [
        "сустав", "воспалительн", "активност", "аутоантител", "срб", "соэ",
        "das28", "базисн", "генно-инженерн",
    ],
    "stomatologiya": [
        "зубн формул", "полост рта", "кариес", "пародонтит", "хирургическ",
        "обезболиван", "профилактик",
    ],
    "travmatologiya-ortopediya": [
        "травм", "перелом", "вывих", "локализац", "рентген", "кт",
        "иммобилизац", "операц", "реабилитац",
    ],
    "transplantatsiya-organov-i-tkaney": [
        "орган", "донор", "реципиент", "иммунологическ", "совместимост",
        "иммуносупресс", "противопоказан", "мониторинг",
    ],
    "urologiya": [
        "мочев", "простат", "почк", "пузыр", "инфекц", "узи", "анализ мочи",
        "пса", "оперативн",
    ],
    "khirurgiya": [
        "острый живот", "хирургическ", "операц", "предоперационн",
        "послеоперационн", "осложнен", "госпитализац",
    ],
    "endokrinologiya-narusheniya-obmena-veshchestv": [
        "сахарный диабет", "гликеми", "hba1c", "щитовидн", "ттг", "т4",
        "ожирен", "метаболическ", "инсулинотерап", "сахароснижающ",
    ],
}

# Человекочитаемые названия рубрик (для отчёта).
RUBRIC_TITLES: dict[str, str] = {
    "akusherstvo-ginekologiya": "Акушерство, гинекология",
    "allergologiya-immunologiya": "Аллергология, иммунология",
    "anesteziologiya-reanimatologiya": "Анестезиология, реаниматология",
    "bolezni-sistemy-krovoobrashcheniya": "Болезни системы кровообращения",
    "gastroenterologiya": "Гастроэнтерология",
    "gematologiya": "Гематология",
    "dermatovenerologiya": "Дерматовенерология",
    "zabolevaniya-perinatalnogo-perioda": "Заболевания перинатального периода",
    "infektsionnye-zabolevaniya": "Инфекционные заболевания",
    "nevrologiya-neyrokhirurgiya": "Неврология, нейрохирургия",
    "nefrologiya": "Нефрология",
    "novoobrazovaniya": "Новообразования",
    "otorinolaringologiya": "Оториноларингология",
    "oftalmologiya": "Офтальмология",
    "palliativnaya-pomoshch": "Паллиативная помощь",
    "psikhiatriya-narkologiya": "Психиатрия, наркология",
    "pulmonologiya-ftiziatriya": "Пульмонология, фтизиатрия",
    "revmatologiya": "Ревматология",
    "stomatologiya": "Стоматология",
    "travmatologiya-ortopediya": "Травматология, ортопедия",
    "transplantatsiya-organov-i-tkaney": "Трансплантация органов и тканей",
    "urologiya": "Урология",
    "khirurgiya": "Хирургия",
    "endokrinologiya-narusheniya-obmena-veshchestv": "Эндокринология, обмен веществ",
}

# --- Числовые/структурные измерения (рубрико-агностично, репортятся при наличии) ---
# Каждый паттерн: (имя, компилированный regex, группа значения, единица).
_MEASUREMENT_SPECS: list[tuple[str, str, int, str]] = [
    ("tnm", r"\b([cp]?T(?:is|[0-4x])\s*N[0-3x]\s*M[01x])\b", 1, ""),
    ("stage", r"стади[июея]\s*([IVX]{1,4}[АВABCС]?|[0-4])\b", 1, ""),
    ("grade", r"\b(G[1-4])\b", 1, ""),
    ("hba1c", r"(?:hba1c|гликирован\w*\s+гемоглобин\w*)\D{0,15}(\d{1,2}[.,]\d)\s*%?", 1, "%"),
    ("glucose", r"(?:глюкоз\w*|гликеми\w*)\D{0,12}(\d{1,2}[.,]\d)\s*(?:ммоль)?", 1, "ммоль/л"),
    ("tsh", r"(?:ттг|тиреотропн\w*)\D{0,12}(\d{1,3}[.,]\d+)", 1, "мМЕ/л"),
    ("creatinine", r"креатинин\D{0,12}(\d{2,4})\s*(?:мкмоль)?", 1, "мкмоль/л"),
    ("gfr", r"(?:скф|gfr|рсkф)\D{0,12}(\d{1,3})\s*(?:мл)?", 1, "мл/мин"),
    ("esr", r"(?:соэ)\D{0,8}(\d{1,3})\s*(?:мм)?", 1, "мм/ч"),
    ("crp", r"(?:срб|c-?реактивн\w*\s*белок|crp)\D{0,12}(\d{1,3}[.,]?\d*)", 1, "мг/л"),
    ("das28", r"das\s*-?\s*28\D{0,6}(\d[.,]\d+)", 1, ""),
    ("fev1", r"(?:офв1|fev1)\D{0,12}(\d{1,3})\s*%?", 1, "%"),
    ("spo2", r"(?:spo2|sao2|сатураци\w*)\D{0,10}(\d{2,3})\s*%", 1, "%"),
    ("iop", r"(?:вгд|внутриглазн\w*\s+давлени\w*)\D{0,12}(\d{1,2})", 1, "мм рт.ст."),
    ("psa", r"(?:пса|psa)\D{0,10}(\d{1,3}[.,]?\d*)\s*(?:нг)?", 1, "нг/мл"),
    ("ejection_fraction", r"(?:фв(?:\s*лж)?|фракци\w*\s+выброс\w*|\bef\b)\D{0,10}(\d{2})\s*%", 1, "%"),
    ("gestational_weeks", r"(?:беремен\w*|гестаци\w*)\D{0,15}(\d{1,2})\s*недел", 1, "нед"),
    ("nihss", r"nihss\D{0,6}(\d{1,2})", 1, "балл"),
    ("birth_weight", r"масс\w*\s+тела\D{0,8}(\d{3,4})\s*г\b", 1, "г"),
]

_COMPILED_MEASUREMENTS = [
    (name, re.compile(pat, re.IGNORECASE | re.UNICODE), grp, unit)
    for name, pat, grp, unit in _MEASUREMENT_SPECS
]


def extract_measurements(text: str) -> dict[str, dict[str, str]]:
    """Извлекает числовые клинические величины. Возвращает {name: {value, unit, raw}}."""
    out: dict[str, dict[str, str]] = {}
    if not text:
        return out
    for name, rx, grp, unit in _COMPILED_MEASUREMENTS:
        try:
            m = rx.search(text)
        except Exception:
            m = None
        if not m:
            continue
        try:
            value = (m.group(grp) or "").strip().replace(",", ".")
        except Exception:
            value = ""
        if not value:
            continue
        out[name] = {"value": value, "unit": unit, "raw": m.group(0).strip()[:80]}
    return out


def normalize_rubric_slug(value: str | None) -> str | None:
    """Приводит произвольный slug/путь к каноническому slug рубрики."""
    if not value:
        return None
    low = value.lower().replace("\\", "/")
    for slug in RUBRIC_TERMS:
        if slug in low:
            return slug
    return None


def rubric_slugs_from_matches(matches: list[dict[str, Any]] | None) -> list[str]:
    """Извлекает уникальные slug рубрик из source_path подобранных протоколов."""
    seen: list[str] = []
    for m in matches or []:
        slug = normalize_rubric_slug((m or {}).get("source_path"))
        if slug and slug not in seen:
            seen.append(slug)
    return seen


def extract_rubric_specifics(
    text: str,
    rubric_slugs: list[str] | None = None,
) -> dict[str, Any]:
    """Главная точка входа: профильные термины по рубрикам + измерения.

    Если rubric_slugs пуст — анализируются все 24 рубрики (для измерений это
    не важно, а term-покрытие считается по найденным рубрикам).
    """
    low = (text or "").lower()
    measurements = extract_measurements(text)

    slugs = [s for s in (rubric_slugs or []) if s in RUBRIC_TERMS]
    # Если рубрики не заданы — берём те, чьи термины реально встречаются (топ-совпадения).
    if not slugs:
        scored: list[tuple[int, str]] = []
        for slug, terms in RUBRIC_TERMS.items():
            hits = sum(1 for t in terms if t in low)
            if hits:
                scored.append((hits, slug))
        scored.sort(key=lambda x: -x[0])
        slugs = [s for _, s in scored[:3]]

    by_rubric: dict[str, Any] = {}
    for slug in slugs:
        terms = RUBRIC_TERMS.get(slug, [])
        matched = [t for t in terms if t in low]
        missing = [t for t in terms if t not in low]
        coverage = round(100.0 * len(matched) / len(terms), 1) if terms else 0.0
        by_rubric[slug] = {
            "title": RUBRIC_TITLES.get(slug, slug),
            "matched_terms": matched,
            "missing_terms": missing,
            "term_coverage_pct": coverage,
        }

    return {
        "rubrics": slugs,
        "by_rubric": by_rubric,
        "measurements": measurements,
    }
