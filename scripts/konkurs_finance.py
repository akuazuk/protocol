"""Единые допущения финмодели конкурса Белинфонд 2026 (МЦ «Кравира»)."""
from __future__ import annotations

# Якорь рынка: 25 000 КЗ/мес в Кравире = 1% платных КЗ частного сектора РБ
# Адресуемый рынок — ВСЕ частные ОЗ Беларуси (~2,5 млн КЗ/мес), не только Кравира
KRAVIRA_KZ_MONTH = 25_000
KRAVIRA_MARKET_SHARE = 0.01
MARKET_KZ_MONTH = int(KRAVIRA_KZ_MONTH / KRAVIRA_MARKET_SHARE)  # 2_500_000
MARKET_KZ_YEAR = MARKET_KZ_MONTH * 12  # 30_000_000

PRICE_START = 0.99
PRICE_CLINIC = 0.79
PRICE_NETWORK = 0.69
PRICE_BLEND_Y2Y3 = 0.75

KRAVIRA_B2B_MONTH = KRAVIRA_KZ_MONTH * PRICE_NETWORK  # 17_250
KRAVIRA_B2B_YEAR = int(KRAVIRA_B2B_MONTH * 12)  # 207_000

# TAM / SAM / SOM (КЗ в год, частный сектор РБ)
TAM_KZ_YEAR = MARKET_KZ_YEAR
SAM_SHARE = 0.05
SAM_KZ_YEAR = int(TAM_KZ_YEAR * SAM_SHARE)  # 1_500_000
SOM_Y3_KZ_MONTH = 200_000
SOM_Y3_KZ_YEAR = SOM_Y3_KZ_MONTH * 12  # 2_400_000
SOM_Y3_MARKET_SHARE = SOM_Y3_KZ_MONTH / MARKET_KZ_MONTH  # 8%

TAM_REVENUE_B2B_YEAR = int(TAM_KZ_YEAR * PRICE_START)  # 29_700_000
TAM_REVENUE_YEAR = TAM_REVENUE_B2B_YEAR  # alias

# --- B2C: все пациенты частных ОЗ РБ (30 млн касаний/год) ---
B2C_TAM_TOUCHES_YEAR = MARKET_KZ_YEAR

B2C_TIERS = [
    {"id": "l1_base", "name": "Базовый L1", "specialty": "терапия, простой приём", "price": 4.99, "mix": 0.38},
    {"id": "l1_plus", "name": "Расширенный L1+", "specialty": "специалист + анализы в КЗ", "price": 6.99, "mix": 0.22},
    {"id": "l2_std", "name": "Стандарт L2", "specialty": "подробный разбор", "price": 9.99, "mix": 0.18},
    {"id": "l2_onko", "name": "Онкология L2", "specialty": "онко, хроника, ЗНО", "price": 14.99, "mix": 0.12},
    {"id": "l2_preop", "name": "Предоперационный L2", "specialty": "pre-op, высокий риск", "price": 12.99, "mix": 0.10},
]

B2C_PROMO_CLINIC = 2.99  # ссылка от клиники, первичный канал


def _b2c_avg_price() -> float:
    return sum(t["price"] * t["mix"] for t in B2C_TIERS)


B2C_AVG_PRICE = round(_b2c_avg_price(), 2)  # ~7,47 BYN

# Rev-share: клиника получает долю при оплате пациентом по ссылке/SMS
CLINIC_B2C_REVSHARE = 0.30  # 30% пациенту видимая скидка / клинике
PROTOCOL_B2C_NET_SHARE = 1.0 - CLINIC_B2C_REVSHARE  # 70% Protocol

# Пример: онкология 14,99 → клинике 4,50, Protocol 10,49
def clinic_revshare_byn(patient_price: float) -> tuple[float, float]:
    clinic = round(patient_price * CLINIC_B2C_REVSHARE, 2)
    protocol = round(patient_price - clinic, 2)
    return clinic, protocol


# Финплан 3 года (тыс. BYN)
# b2b_k включает Кравиру (207) + другие клиники; b2c_k — национальный B2C-поток
FIN_Y1 = {
    "clients": 1,
    "kz_month": 25_000,
    "b2b_k": 207,
    "b2b_kravira_k": 207,
    "b2b_other_k": 0,
    "b2c_k": 50,
    "b2c_checks": 6_700,
    "opex_k": 280,
    "api_k": 0,
}
FIN_Y2 = {
    "clients": 3,
    "kz_month": 75_000,
    "b2b_k": 675,
    "b2b_kravira_k": 207,
    "b2b_other_k": 468,
    "b2c_k": 180,
    "b2c_checks": 24_100,
    "opex_k": 420,
    "api_k": 25,
}
FIN_Y3 = {
    "clients": 8,
    "kz_month": 200_000,
    "b2b_k": 1_800,
    "b2b_kravira_k": 207,
    "b2b_other_k": 1_593,
    "b2c_k": 520,
    "b2c_checks": 69_600,
    "opex_k": 650,
    "api_k": 75,
}

# B2C upside при конверсии 0,3% TAM (~90k проверок/год × ~7,5 BYN ≈ 675k BYN) — в таблице сценариев
B2C_UPSIDE_YEAR3_K = 675

# year 2: 3% рынка; year 3: 8% рынка (B2B поток)
Y2_MARKET_SHARE = FIN_Y2["kz_month"] / MARKET_KZ_MONTH
Y3_MARKET_SHARE = FIN_Y3["kz_month"] / MARKET_KZ_MONTH

# B2C конверсия от TAM
B2C_CONV_Y1 = FIN_Y1["b2c_checks"] / B2C_TAM_TOUCHES_YEAR
B2C_CONV_Y3 = FIN_Y3["b2c_checks"] / B2C_TAM_TOUCHES_YEAR

# Доход клиник-партнёров от rev-share B2C (тыс. BYN) — доп. мотивация для ОЗ
def clinic_b2c_revshare_k(b2c_protocol_k: int) -> int:
    """Если b2c_k — выручка Protocol (70%), gross patient = b2c / 0.7, clinic share 30%."""
    gross = b2c_protocol_k / PROTOCOL_B2C_NET_SHARE
    return int(gross * CLINIC_B2C_REVSHARE)


CLINIC_B2C_REV_Y1_K = clinic_b2c_revshare_k(FIN_Y1["b2c_k"])
CLINIC_B2C_REV_Y2_K = clinic_b2c_revshare_k(FIN_Y2["b2c_k"])
CLINIC_B2C_REV_Y3_K = clinic_b2c_revshare_k(FIN_Y3["b2c_k"])

# Сертификат ГКНТ
CERTIFICATE_BV = 571
BASE_VALUE_BYN = 42
CERTIFICATE_BYN = CERTIFICATE_BV * BASE_VALUE_BYN  # 23_982

# ROI Кравira
ROI_PROTOCOL_COST = KRAVIRA_B2B_MONTH
ROI_METHODIST_FTE_SAVED = 0.35
ROI_METHODIST_FTE_COST = 3_200
ROI_METHODIST_SAVING = int(ROI_METHODIST_FTE_SAVED * ROI_METHODIST_FTE_COST)
ROI_CISZ_REJECT_RATE_BASE = 0.04
ROI_CISZ_REJECT_RATE_AFTER = 0.025
ROI_CISZ_REWORK_MINUTES = 25
ROI_CISZ_MINUTE_COST = 2.5
ROI_CISZ_SAVING = int(
    KRAVIRA_KZ_MONTH
    * (ROI_CISZ_REJECT_RATE_BASE - ROI_CISZ_REJECT_RATE_AFTER)
    * ROI_CISZ_REWORK_MINUTES
    * ROI_CISZ_MINUTE_COST
)
ROI_TOTAL_SAVING = ROI_METHODIST_SAVING + ROI_CISZ_SAVING
ROI_NET = ROI_TOTAL_SAVING - ROI_PROTOCOL_COST


def ebitda_k(year: dict) -> int:
    rev = year["b2b_k"] + year["b2c_k"] + year.get("api_k", 0)
    return rev - year["opex_k"]


def total_rev_k(year: dict) -> int:
    return year["b2b_k"] + year["b2c_k"] + year.get("api_k", 0)
