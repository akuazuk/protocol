"""Дополнительные каналы монетизации Protocol (расширение базового плана)."""
from __future__ import annotations

from konkurs_finance import FIN_Y3, total_rev_k, ebitda_k
from konkurs_scenarios import SCENARIO_BASE, SCENARIO_OPTIMISTIC

# Доп. потоки к 2029 (тыс. BYN/год) — поверх осторожного B2B+B2C+API
EXTRA_STREAMS_Y3 = [
    {
        "name": "B2B L2 для методиста",
        "price": "+0,50 BYN/КЗ",
        "y3_k": 180,
        "driver": "15% потока B2B (200 тыс. КЗ/мес) с доп. разбором",
        "prob": 0.75,
        "phase": "2028",
    },
    {
        "name": "Подписка «Методслужба Pro»",
        "price": "25 000 BYN/год",
        "y3_k": 200,
        "driver": "Дашборд, квартальный аудит, 8 ОЗ",
        "prob": 0.65,
        "phase": "2028",
    },
    {
        "name": "Обучение врачей (курсы КП)",
        "price": "800–1 500 BYN/место",
        "y3_k": 120,
        "driver": "12 сессий × 10–15 врачей, очно/онлайн",
        "prob": 0.55,
        "phase": "2027",
    },
    {
        "name": "OEM/API премиум (МИС)",
        "price": "50–120 тыс. внедрение",
        "y3_k": 125,
        "driver": "Доп. к базовым 75 тыс.: 2-й вендор + SLA",
        "prob": 0.52,
        "phase": "2029",
    },
    {
        "name": "B2C медтуризм",
        "price": "19,99 BYN/проверка",
        "y3_k": 95,
        "driver": "~8 500 проверок иностранных пациентов",
        "prob": 0.45,
        "phase": "2029",
    },
    {
        "name": "Корпоративный аудит КЗ",
        "price": "60–100 тыс./договор",
        "y3_k": 160,
        "driver": "Страховщики, ДМС, корп. клиенты клиник",
        "prob": 0.40,
        "phase": "2029",
    },
    {
        "name": "White-label Enterprise",
        "price": "8 000 BYN/мес",
        "y3_k": 192,
        "driver": "2 сети ОЗ: свой бренд B2C + API",
        "prob": 0.35,
        "phase": "2029",
    },
    {
        "name": "Аналитика для руководства ОЗ",
        "price": "12 000 BYN/мес",
        "y3_k": 144,
        "driver": "Агрегированные KPI качества КЗ (без ПДн)",
        "prob": 0.50,
        "phase": "2029",
    },
]

EXTRA_REV_Y3_K = sum(s["y3_k"] for s in EXTRA_STREAMS_Y3)  # 1 216
EXTRA_OPEX_Y3_K = 280  # ФОТ: enterprise-продажи, тренеры, поддержка OEM

BASE_REV_Y3_K = total_rev_k(FIN_Y3)
BASE_EBITDA_Y3_K = ebitda_k(FIN_Y3)

EXPANDED_Y3 = {
    "label": "Расширенный (базовый + доп. каналы)",
    "base_rev_k": BASE_REV_Y3_K,
    "extra_rev_k": EXTRA_REV_Y3_K,
    "total_rev_k": BASE_REV_Y3_K + EXTRA_REV_Y3_K,
    "opex_k": FIN_Y3["opex_k"] + EXTRA_OPEX_Y3_K,
    "ebitda_k": BASE_REV_Y3_K + EXTRA_REV_Y3_K - FIN_Y3["opex_k"] - EXTRA_OPEX_Y3_K,
    "ebitda_month_k": round((BASE_REV_Y3_K + EXTRA_REV_Y3_K - FIN_Y3["opex_k"] - EXTRA_OPEX_Y3_K) / 12, 1),
    "note": "Осторожный B2B/B2C + 8 доп. потоков к 2029",
}

# Сравнение всех сценариев Y3 для таблицы
ALL_REVENUE_SCENARIOS_Y3 = [
    ("Осторожный", BASE_REV_Y3_K, BASE_EBITDA_Y3_K, "8% TAM B2B"),
    ("Базовый", SCENARIO_BASE["total_rev_k"], SCENARIO_BASE["ebitda_k"], "10% TAM"),
    ("Оптимистичный", SCENARIO_OPTIMISTIC["total_rev_k"], SCENARIO_OPTIMISTIC["ebitda_k"], "12% TAM"),
    ("Расширенный", EXPANDED_Y3["total_rev_k"], EXPANDED_Y3["ebitda_k"], "базовый + 8 каналов"),
]

MONETIZATION_TABLE = [
    (s["name"], s["price"], f"{s['y3_k']:,}".replace(",", " "), s["driver"], f"{s['prob']:.0%}", s["phase"])
    for s in EXTRA_STREAMS_Y3
]

MONETIZATION_INTRO = """
Базовый план (B2B микроплатёж + B2C tier + API) — консервативный фундамент. Ниже — восемь дополнительных каналов,
которые естественно вытекают из продукта: методслужба, обучение, OEM, медтуризм, корпоративные аудиты.
Они не требуют нового ядра — только упаковку, продажи и договоры. Суммарный upside к 2029: +1,2 млн BYN/год
к осторожному сценарию (выручка до ~3,5 млн, EBITDA до ~2,5 млн/год ≈ 205 тыс./мес).
""".strip()

WEIGHTED_EXTRA_Y3_K = sum(int(s["prob"] * s["y3_k"]) for s in EXTRA_STREAMS_Y3)
