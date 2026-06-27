"""Сценарии монетизации и прогноз каналов (Белинфонд 2026)."""
from __future__ import annotations

from konkurs_finance import (
    B2C_AVG_PRICE,
    B2C_TAM_TOUCHES_YEAR,
    FIN_Y1,
    FIN_Y2,
    FIN_Y3,
    KRAVIRA_B2B_YEAR,
    MARKET_KZ_MONTH,
    MARKET_KZ_YEAR,
    PRICE_BLEND_Y2Y3,
    PROTOCOL_B2C_NET_SHARE,
    SAM_KZ_YEAR,
    SOM_Y3_KZ_MONTH,
    SOM_Y3_KZ_YEAR,
    TAM_REVENUE_B2B_YEAR,
    ebitda_k,
    total_rev_k,
)

# Доля B2C без rev-share (прямой SEO / protocol.by) - Protocol получает 100% цены
B2C_DIRECT_SHARE = 0.20


def b2c_protocol_byn_per_check(direct_share: float = B2C_DIRECT_SHARE) -> float:
    """Средняя выручка Protocol с одной B2C-проверки (микс rev-share + прямой канал)."""
    via_clinic = (1.0 - direct_share) * B2C_AVG_PRICE * PROTOCOL_B2C_NET_SHARE
    direct = direct_share * B2C_AVG_PRICE
    return round(via_clinic + direct, 3)


B2C_PROTOCOL_PER_CHECK = b2c_protocol_byn_per_check()


def b2c_protocol_k(checks_year: int, direct_share: float = B2C_DIRECT_SHARE) -> int:
    return int(checks_year * b2c_protocol_byn_per_check(direct_share) / 1000)


def b2b_k_from_kz_month(kz_month: int) -> int:
    return int(kz_month * 12 * PRICE_BLEND_Y2Y3 / 1000)


def checks_from_b2c_conv(conv: float) -> int:
    return int(B2C_TAM_TOUCHES_YEAR * conv)


def conv_from_checks(checks: int) -> float:
    return checks / B2C_TAM_TOUCHES_YEAR


# --- Прогноз каналов (вероятность успеха к 2029, экспертная оценка) ---
CHANNEL_OUTLOOK = [
    {
        "channel": "B2B якорь (Кравира)",
        "prob": 0.95,
        "y3_k": KRAVIRA_B2B_YEAR // 1000,
        "driver": "Пилот уже в production; фикс. контракт 0,69 BYN/КЗ",
        "rank": 1,
    },
    {
        "channel": "B2B прямые клиники",
        "prob": 0.72,
        "y3_k": FIN_Y3["b2b_other_k"],
        "driver": "Давление ЦИСЗ + ROI кейс; 7 ОЗ кроме якоря",
        "rank": 2,
    },
    {
        "channel": "B2C SMS/QR rev-share",
        "prob": 0.68,
        "y3_k": int(FIN_Y3["b2c_k"] * 0.75),
        "driver": "30% клинике - мотивация рассылать ссылку после приёма",
        "rank": 3,
    },
    {
        "channel": "B2B API / OEM (Айболит)",
        "prob": 0.52,
        "y3_k": FIN_Y3["api_k"],
        "driver": "Масштаб через вендора МИС; цикл продаж 12-18 мес",
        "rank": 4,
    },
    {
        "channel": "B2C SEO / прямой вход",
        "prob": 0.41,
        "y3_k": int(FIN_Y3["b2c_k"] * 0.25),
        "driver": "Длинный хвост; без rev-share, выше маржа Protocol",
        "rank": 5,
    },
]

# Взвешенный прогноз выручки по каналам (prob × y3_k)
WEIGHTED_CHANNEL_Y3_K = sum(int(c["prob"] * c["y3_k"]) for c in CHANNEL_OUTLOOK)

# --- Три сценария года 3 (2029) ---
# EBITDA считается от доли рынка B2B (SOM), не от полного TAM 2,5 млн КЗ/мес
SCENARIOS_Y3 = {
    "cautious": {
        "label": "Осторожный (базовый план)",
        "b2b_share": 0.08,
        "kz_month": 200_000,
        "b2c_conv": 0.00232,
        "api_k": 75,
        "opex_k": 650,
        "note": "8% TAM B2B · 0,23% B2C · 8 клиник",
    },
    "base": {
        "label": "Базовый (вероятный)",
        "b2b_share": 0.10,
        "kz_month": 250_000,
        "b2c_conv": 0.0033,
        "api_k": 100,
        "opex_k": 720,
        "note": "10% TAM B2B · 0,33% B2C · SMS rev-share масштабируется",
    },
    "optimistic": {
        "label": "Оптимистичный (upside)",
        "b2b_share": 0.12,
        "kz_month": 300_000,
        "b2c_conv": 0.005,
        "api_k": 150,
        "opex_k": 850,
        "note": "12% TAM B2B · 0,5% B2C · OEM API + сети ОЗ",
    },
}


def build_scenario_y3(key: str) -> dict:
    s = SCENARIOS_Y3[key]
    kz = s["kz_month"]
    checks = checks_from_b2c_conv(s["b2c_conv"])
    b2b_k = b2b_k_from_kz_month(kz)
    b2c_k = b2c_protocol_k(checks)
    api_k = s["api_k"]
    opex_k = s["opex_k"]
    rev = b2b_k + b2c_k + api_k
    ebitda = rev - opex_k
    return {
        **s,
        "key": key,
        "b2b_k": b2b_k,
        "b2b_kravira_k": KRAVIRA_B2B_YEAR // 1000,
        "b2b_other_k": b2b_k - KRAVIRA_B2B_YEAR // 1000,
        "b2c_k": b2c_k,
        "b2c_checks": checks,
        "total_rev_k": rev,
        "ebitda_k": ebitda,
        "ebitda_month_k": round(ebitda / 12, 1),
        "rev_month_k": round(rev / 12, 1),
    }


SCENARIO_CAUTIOUS = build_scenario_y3("cautious")
SCENARIO_BASE = build_scenario_y3("base")
SCENARIO_OPTIMISTIC = build_scenario_y3("optimistic")
ALL_SCENARIOS_Y3 = [SCENARIO_CAUTIOUS, SCENARIO_BASE, SCENARIO_OPTIMISTIC]

# Теоретический потолок (100% TAM) - для графика, не план
TAM_CEILING_B2B_YEAR_K = int(MARKET_KZ_YEAR * PRICE_BLEND_Y2Y3 / 1000)  # ~22 500
TAM_CEILING_B2C_YEAR_K = b2c_protocol_k(int(B2C_TAM_TOUCHES_YEAR * 0.01))  # 1% conv


# TAM → SAM → SOM → выручка (мост для графика)
TAM_BRIDGE = [
    ("TAM - весь рынок", MARKET_KZ_YEAR, TAM_CEILING_B2B_YEAR_K, "30 млн КЗ/год · теор. B2B 22 500 тыс. BYN/год"),
    ("SAM - крупные ОЗ 5%", SAM_KZ_YEAR, int(SAM_KZ_YEAR * PRICE_BLEND_Y2Y3 / 1000), "1,5 млн КЗ/год · целевой B2B-сегмент"),
    ("SOM - план 2029 (8%)", SOM_Y3_KZ_YEAR, SCENARIO_CAUTIOUS["b2b_k"], "2,4 млн КЗ/год · 200 тыс. КЗ/мес B2B"),
    ("B2C + API 2029", SCENARIO_CAUTIOUS["b2c_checks"], SCENARIO_CAUTIOUS["b2c_k"] + SCENARIO_CAUTIOUS["api_k"], "69,6 тыс. проверок · 0,23% частного TAM B2C"),
    ("Выручка Protocol", 0, SCENARIO_CAUTIOUS["total_rev_k"], "осторожный сценарий · 2 315 тыс. BYN/год"),
]

# Чувствительность EBITDA к доле рынка B2B (B2C фикс. как в cautious)
def ebitda_sensitivity_b2b_share(share: float) -> int:
    kz = int(MARKET_KZ_MONTH * share)
    b2b = b2b_k_from_kz_month(kz)
    b2c = SCENARIO_CAUTIOUS["b2c_k"]
    api = SCENARIO_CAUTIOUS["api_k"]
    opex = SCENARIO_CAUTIOUS["opex_k"]
    return b2b + b2c + api - opex


PENETRATION_SENSITIVITY = [
    (pct, ebitda_sensitivity_b2b_share(pct / 100))
    for pct in (3, 5, 8, 10, 12, 15, 20)
]

# Чувствительность EBITDA к конверсии B2C (B2B фикс. 8% TAM частного сектора, API и OPEX - cautious)
def ebitda_sensitivity_b2c_conv(conv: float, b2b_share: float = 0.08) -> int:
    from konkurs_finance import MARKET_KZ_MONTH

    kz = int(MARKET_KZ_MONTH * b2b_share)
    b2b = b2b_k_from_kz_month(kz)
    checks = int(B2C_TAM_TOUCHES_YEAR * conv)
    b2c = b2c_protocol_k(checks)
    api = SCENARIO_CAUTIOUS["api_k"]
    opex = SCENARIO_CAUTIOUS["opex_k"]
    return b2b + b2c + api - opex


def build_b2c_conv_sensitivity_rows() -> list[tuple[str, str, str, str, str]]:
    """conv, checks, b2c_k, total rev, EBITDA (B2B 8% и OPEX фикс.)."""
    b2b_k = b2b_k_from_kz_month(int(MARKET_KZ_MONTH * 0.08))
    api_k = SCENARIO_CAUTIOUS["api_k"]
    opex_k = SCENARIO_CAUTIOUS["opex_k"]

    levels: list[tuple[float | None, int | None]] = [
        (0.001, None),
        (None, FIN_Y3["b2c_checks"]),
        (0.0033, None),
        (0.005, None),
        (0.01, None),
    ]
    rows: list[tuple[str, str, str, str, str]] = []
    for conv, checks_override in levels:
        if checks_override is not None:
            checks = checks_override
            conv_f = checks / B2C_TAM_TOUCHES_YEAR
        else:
            assert conv is not None
            checks = int(B2C_TAM_TOUCHES_YEAR * conv)
            conv_f = conv
        b2c_k = b2c_protocol_k(checks)
        rev_k = b2b_k + b2c_k + api_k
        ebitda = rev_k - opex_k
        conv_pct = f"{conv_f * 100:.3f}".replace(".", ",").rstrip("0").rstrip(",") + "%"
        rows.append(
            (
                conv_pct,
                f"{checks:,}".replace(",", " "),
                str(b2c_k),
                str(rev_k),
                str(ebitda),
            )
        )
    return rows


B2C_CONV_SENSITIVITY = build_b2c_conv_sensitivity_rows()

# Синхронизация FIN_Y* с исправленной B2C-формулой
def _sync_fin_year(year: dict, checks: int) -> dict:
    y = dict(year)
    y["b2c_checks"] = checks
    y["b2c_k"] = b2c_protocol_k(checks)
    return y


FIN_Y1_SYNC = _sync_fin_year(FIN_Y1, FIN_Y1["b2c_checks"])
FIN_Y2_SYNC = _sync_fin_year(FIN_Y2, FIN_Y2["b2c_checks"])
FIN_Y3_SYNC = _sync_fin_year(FIN_Y3, FIN_Y3["b2c_checks"])

# Таблица для HTML: сравнение сценариев Y3
SCENARIO_COMPARE_TABLE = [
    (
        s["label"],
        f"{s['b2b_share']:.0%}",
        f"{s['b2c_conv']:.2%}".replace(".", ","),
        f"{s['total_rev_k']:,}".replace(",", " "),
        f"+{s['ebitda_k']:,}".replace(",", " "),
        f"+{s['ebitda_month_k']:.0f}".replace(".", ","),
    )
    for s in ALL_SCENARIOS_Y3
]

CHANNEL_TABLE = [
    (c["channel"], f"{c['prob']:.0%}", f"{c['y3_k']:,}".replace(",", " "), c["driver"])
    for c in sorted(CHANNEL_OUTLOOK, key=lambda x: x["rank"])
]

MONTHLY_Y3_CAUTIOUS = {
    "rev_k": round(total_rev_k(FIN_Y3_SYNC) / 12, 1),
    "ebitda_k": round(ebitda_k(FIN_Y3_SYNC) / 12, 1),
    "b2b_k": round(FIN_Y3_SYNC["b2b_k"] / 12, 1),
    "b2c_k": round(FIN_Y3_SYNC["b2c_k"] / 12, 1),
    "kz_month": FIN_Y3_SYNC["kz_month"],
    "tam_share_pct": FIN_Y3_SYNC["kz_month"] / MARKET_KZ_MONTH * 100,
}
