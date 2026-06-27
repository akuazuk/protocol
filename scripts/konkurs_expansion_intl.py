"""Международная экспансия Protocol: рынки с гос. регламентами, TAM и GTM (§6.3 БП)."""
from __future__ import annotations

from dataclasses import dataclass

from konkurs_scenarios import B2C_PROTOCOL_PER_CHECK, b2c_protocol_k

# --- Допущения unit economics (не в EBITDA 2029 базового плана) ---
BYN_PER_USD = 3.2
B2C_REV_CIS_BYN = B2C_PROTOCOL_PER_CHECK  # 6,331 - микс tier + rev-share, как в РБ
B2C_REV_INTL_BYN = 8.0  # ~2,5 USD экв.; tier 4,99-14,99 в локальной валюте

# Конверсии: «осторожная» и «базовая» для upside-таблицы (не финплан 2029)
CONV_CAUTIOUS = 0.0005  # 0,05%
CONV_BASE_CIS = 0.001  # 0,10%
CONV_BASE_EN = 0.0003  # 0,03%


@dataclass(frozen=True)
class IntlMarket:
    country: str
    tier: str
    population_m: float
    visits_year_m: int
    corpus_label: str
    corpus_url: str
    language: str
    launch: str
    conv_cautious: float
    conv_base: float
    rev_per_check_byn: float
    b2b_horizon: str
    entry_cost_k: int  # тыс. BYN, год 1


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def intl_checks(visits: int, conv: float) -> int:
    return int(visits * conv)


def intl_b2c_rev_k(checks: int, per_check_byn: float) -> int:
    return int(checks * per_check_byn / 1000)


def _market_row(m: IntlMarket, conv: float) -> tuple[str, ...]:
    visits = m.visits_year_m * 1_000_000
    checks = intl_checks(visits, conv)
    rev = intl_b2c_rev_k(checks, m.rev_per_check_byn)
    conv_pct = f"{conv * 100:.3f}".replace(".", ",").rstrip("0").rstrip(",") + "%"
    return (
        m.country,
        f"Tier {m.tier}",
        _fmt_int(m.visits_year_m) + " млн",
        conv_pct,
        _fmt_int(checks),
        str(rev),
        m.launch,
    )


INTERNATIONAL_MARKETS: list[IntlMarket] = [
    IntlMarket(
        "Россия",
        "1",
        146,
        550,
        "700+ КР Минздрава",
        "cr.minzdrav.gov.ru",
        "RU",
        "2030-2031",
        CONV_CAUTIOUS,
        CONV_BASE_CIS,
        B2C_REV_CIS_BYN,
        "2032+ через партнёра МИС",
        85,
    ),
    IntlMarket(
        "Казахстан",
        "1",
        20,
        45,
        "КП МЗ РК (rcrz.kz / pdl.kz)",
        "pdl.kz",
        "RU/KZ",
        "2030",
        0.001,
        0.00232,
        B2C_REV_CIS_BYN,
        "2031 частные сети",
        55,
    ),
    IntlMarket(
        "Узбекистан",
        "1",
        37,
        85,
        "Клин. протоколы МЗ (2020-2026)",
        "MedElement / МЗ",
        "RU/UZ",
        "2030-2031",
        0.0005,
        0.001,
        B2C_REV_CIS_BYN,
        "2032+",
        50,
    ),
    IntlMarket(
        "Киргизия",
        "1",
        7,
        16,
        "КП МЗ КР",
        "MedElement",
        "RU",
        "2031",
        0.001,
        0.00232,
        B2C_REV_CIS_BYN,
        "2032+",
        35,
    ),
    IntlMarket(
        "Индия",
        "2",
        1400,
        1500,
        "STG/STW ICMR + MOHFW",
        "stw.icmr.org.in",
        "EN",
        "2031-2032",
        0.0001,
        CONV_BASE_EN,
        B2C_REV_INTL_BYN,
        "2033+ ABHA/MIS",
        120,
    ),
    IntlMarket(
        "Турция",
        "2",
        85,
        220,
        "Klinik rehber/protokol",
        "shgmargestddb.saglik.gov.tr",
        "TR",
        "2031-2032",
        0.0002,
        0.0005,
        B2C_REV_INTL_BYN,
        "2033+",
        95,
    ),
    IntlMarket(
        "Египет",
        "2",
        110,
        130,
        "EHC national CPG (49+)",
        "lms.ehc.gov.eg",
        "AR/EN",
        "2032",
        0.0002,
        0.0008,
        B2C_REV_INTL_BYN,
        "2033+",
        90,
    ),
    IntlMarket(
        "Бразилия",
        "2",
        215,
        420,
        "PCDT CONITEC / SUS",
        "gov.br/conitec",
        "PT",
        "2032-2033",
        0.0002,
        0.0005,
        B2C_REV_INTL_BYN,
        "2033+ SUS/particulares",
        110,
    ),
    IntlMarket(
        "Филиппины",
        "2",
        115,
        140,
        "DOH-approved CPG",
        "doh.gov.ph",
        "EN",
        "2032-2033",
        0.0002,
        0.0008,
        B2C_REV_INTL_BYN,
        "2033+",
        75,
    ),
]

# Таблица: осторожная конверсия
INTL_MARKET_CAUTIOUS_TABLE = [_market_row(m, m.conv_cautious) for m in INTERNATIONAL_MARKETS]

# Таблица: базовая конверсия (upside)
INTL_MARKET_BASE_TABLE = [_market_row(m, m.conv_base) for m in INTERNATIONAL_MARKETS]


def _sum_rev(table: list[tuple[str, ...]]) -> int:
    return sum(int(row[5]) for row in table)


INTL_UPSIDE_CAUTIOUS_K = _sum_rev(INTL_MARKET_CAUTIOUS_TABLE)
INTL_UPSIDE_BASE_K = _sum_rev(INTL_MARKET_BASE_TABLE)

# Сумма только Tier 1 (CIS) при базовой конверсии - первый волновой фронт
INTL_TIER1_BASE_K = sum(
    intl_b2c_rev_k(intl_checks(m.visits_year_m * 1_000_000, m.conv_base), m.rev_per_check_byn)
    for m in INTERNATIONAL_MARKETS
    if m.tier == "1"
)

INTL_ENTRY_COST_TOTAL_K = sum(m.entry_cost_k for m in INTERNATIONAL_MARKETS[:5])  # первые 5 рынков к 2032

_IN_B2C_K = intl_b2c_rev_k(intl_checks(1_500_000_000, CONV_BASE_EN), B2C_REV_INTL_BYN)

GTM_PHASES = [
    (
        "A · 2029-2030",
        "РБ",
        "B2B 8% TAM + B2C 0,23%; корпус 478 КП; ROI Кравира",
        "EBITDA +1 665 тыс. BYN (базовый план)",
        "0",
    ),
    (
        "B · 2030-2031",
        "Tier 1: KZ, UZ, РФ",
        "Ingest корпуса; locale RU; B2C SEO; партнёр-резидент РФ",
        f"B2C upside Tier 1 ~{INTL_TIER1_BASE_K:,} тыс. BYN/год".replace(",", " "),
        "190",
    ),
    (
        "C · 2031-2032",
        "Tier 2: IN, TR",
        "EN/TR embedder; UPI/Mir; пилот 1 000 PDF",
        f"+{_IN_B2C_K:,} тыс. BYN только Индия (0,03%)".replace(",", " "),
        "215",
    ),
    (
        "D · 2032-2033",
        "EG, BR, PH",
        "PT/AR packs; CONITEC/EHC ingest",
        f"Суммарный intl upside ~{INTL_UPSIDE_BASE_K:,} тыс. BYN/год".replace(",", " "),
        "275",
    ),
]

GTM_STEPS = [
    (
        "1. Аудит корпуса",
        "500-1 000 официальных PDF целевой страны; сравнение структуры с КП РБ",
        "2-4 нед.",
        "Go/no-go по покрытию нозологий",
    ),
    (
        "2. Legal & data",
        "Резидент / DPA; хостинг in-country или on-prem; минимизация ПДн",
        "1-3 мес.",
        "152-ФЗ (РФ), LGPD (BR), KVKK (TR), DPDP (IN)",
    ),
    (
        "3. Locale ML pack",
        "Fine-tune embedder/reranker; каталог send_gate; дисклеймеры UI",
        "1-2 мес.",
        "ml/datasets/, export_training_feedback.py",
    ),
    (
        "4. B2C pilot",
        "patient.html white-label; 200-500 реальных PDF; NPS пациентов",
        "1 мес.",
        "Без B2B / без МИС",
    ),
    (
        "5. Клиники rev-share",
        "QR/SMS 30/70; 1-3 якорные частные ОЗ",
        "2-3 мес.",
        "Flywheel как в §3.1",
    ),
    (
        "6. SEO & pricing",
        "Локальные tier-цены; «проверить заключение врача»",
        "ongoing",
        "CAC < 1 проверки",
    ),
    (
        "7. B2B L0",
        "Интегратор МИС; pre-sign gate в потоке КЗ",
        "6-18 мес.",
        "После доказанного B2C",
    ),
]

INTL_CORPUS_TABLE = [
    (m.country, m.corpus_label, m.corpus_url, m.language, m.tier)
    for m in INTERNATIONAL_MARKETS
]

INTL_ENTRY_COST_TABLE = [
    (m.country, str(m.entry_cost_k), m.launch, m.b2b_horizon)
    for m in INTERNATIONAL_MARKETS
]

CALC_METHODOLOGY = """
Методика расчёта международного B2C-upside (§6.3, не в EBITDA 2029).

1) Знаменатель TAM - амбулаторные обращения/год (млн), оценка: население × 1,1-1,6 визита/год или отраслевые статистики (РФ ~550 млн; Индия ~1,5 млрд STG-addressable outpatient).

2) Проверок/год = TAM × конверсия. Конверсия осторожная (0,05-0,10% для CIS, 0,01-0,03% для EN) - доля пациентов, загрузивших PDF на patient.html.

3) Выручка Protocol (тыс. BYN) = проверок × цена Protocol/проверка / 1000. Tier 1 (RU/KZ/UZ/KG): 6,331 BYN (как РБ). Tier 2 (IN/TR/EG/BR/PH): 8,0 BYN (~2,5 USD) - tier 4,99-14,99 в локальной валюте.

4) B2B и OPEX страны не включены - таблица показывает только B2C Protocol. Полный P&L страны = B2C + B2B (с 2032+) − локальный OPEX (15-25% от выручки при масштабе).

5) Суммарный upside всех 9 рынков (базовая conv.): {base_k} тыс. BYN/год; осторожная conv.: {caut_k} тыс. BYN/год. Инвестиции входа (5 рынков к 2032): ~{entry_k} тыс. BYN единовременно + 15-20% годового OPEX локальных команд.
""".strip().format(
    base_k=_fmt_int(INTL_UPSIDE_BASE_K),
    caut_k=_fmt_int(INTL_UPSIDE_CAUTIOUS_K),
    entry_k=_fmt_int(INTL_ENTRY_COST_TOTAL_K),
)

INTL_EXPANSION_INTRO = """
Protocol масштабируется на рынки, где государство публикует **обязательные или квази-обязательные** клинические протоколы, клинические рекомендации (КР), STG/PCDT или clinical pathways в открытом доступе. Продукт не привязан к юрисдикции РБ: меняется корпус PDF, языковой pack ML и конфигурация send_gate; архитектура RAG + evidence_map + scoring сохраняется.

Критерий отбора страны: (1) официальный репозиторий регламентов; (2) структура «диагностика - лечение - мониторинг»; (3) пациент или клиника может сопоставить выписку/КЗ с этим стандартом; (4) достаточный амбулаторный поток для B2C.

Три tier: **Tier 1 (CIS)** - клинические протоколы/КР на русском, минимальная стоимость входа после РБ. **Tier 2** - крупные рынки с STG/CPG/PCDT на EN/PT/TR/AR. **Tier 3** - **США (§6.5)** с payer/Epic economics; EU/NICE и Китай - после US playbook или партнёрства.
""".strip()

INTL_TIER1 = """
**Tier 1 - русскоязычный кластер (2030-2031).** Россия: рубрикатор cr.minzdrav.gov.ru, 700+ КР; юридически - партнёр-резидент, 152-ФЗ, Mir/СБП. Казахстан: обязательные КП на pdl.kz / rcrz.kz; ЕАЭС упрощает контур. Узбекистан и Кырgyzstan: протоколы МЗ, русский в меддокументах; вход через B2C SEO и 1-2 частные клиники rev-share.

Почему первыми: один ingest-пайплайн (как minzdrav_protocols/), русский UI и ML уже в production, те же tier-цены и дисклеймеры patient.html.
""".strip()

INTL_TIER2 = """
**Tier 2 - англоязычный и LATAM (2031-2033).** Индия: STG/STW ICMR + MOHFW (clinicalestablishments.gov.in) - привязка к Clinical Establishments Act; UPI-платежи; EN embedder. Турция: klinik rehber на shgmargestddb.saglik.gov.tr, PDF на dosyamerkez.saglik.gov.tr. Египет: Egyptian Health Council (lms.ehc.gov.eg), программа с WHO, 49+ CPG к 2026. Бразилия: PCDT CONITEC + открытые метаданные dadosabertos.saude.gov.br. Филиппины: DOH-approved CPG (AGREE II).

Вход: B2C «Does my discharge summary match national STG/PCDT?» без МИС; B2B - после 500+ проверок и якорной клиники.
""".strip()

INTL_FLYWHEEL = """
Международный flywheel повторяет §3.1: **B2C** (пациент проверяет выписку) → давление на **частные клиники** подключить **B2B L0** → методист и регулятор получают агрегированную аналитику по пробелам в исполнении регламентов → корпус и ML дообучаются на локальном feedback.

Отличие от РБ: в gos-sector B2C запускается раньше B2B (нет ЦИСЗ-интеграции); monetization сначала direct + rev-share, затем API/OEM.
""".strip()

INTL_RISKS = [
    ("Трансграничные ПДн", "Средний", "In-country processing, DPA, on-prem B2B"),
    ("Языковой drift ML", "Средний", "Locale pack + 500 PDF pilot gate"),
    ("Фрагментация корпуса (EU/US)", "Высокий", "Фокус Tier 1-2 с единым MOH portal"),
    ("Локальные CDSS/LLM", "Средний", "N регламентов с цитатами, send_gate"),
    ("FX и платежи", "Низкий", "Локальные acquirer; tier в местной валюте"),
]
