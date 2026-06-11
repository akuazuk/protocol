"""Единые допущения финмодели конкурса Белинфонд 2026 (МЦ «Кравира»)."""
from __future__ import annotations

# Якорь рынка: 25 000 КЗ/мес = 1% платных КЗ частного сектора РБ
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

# TAM / SAM / SOM (КЗ в год)
TAM_KZ_YEAR = MARKET_KZ_YEAR
SAM_SHARE = 0.05
SAM_KZ_YEAR = int(TAM_KZ_YEAR * SAM_SHARE)  # 1_500_000
SOM_Y3_KZ_MONTH = 200_000
SOM_Y3_KZ_YEAR = SOM_Y3_KZ_MONTH * 12  # 2_400_000
SOM_Y3_MARKET_SHARE = SOM_Y3_KZ_MONTH / MARKET_KZ_MONTH  # 8%

TAM_REVENUE_YEAR = int(TAM_KZ_YEAR * PRICE_START)  # 29_700_000

# Финплан 3 года (тыс. BYN)
FIN_Y1 = {"clients": 1, "kz_month": 25_000, "b2b_k": 207, "b2c_k": 50, "opex_k": 280, "api_k": 0}
FIN_Y2 = {"clients": 3, "kz_month": 75_000, "b2b_k": 675, "b2c_k": 180, "opex_k": 420, "api_k": 25}
FIN_Y3 = {"clients": 8, "kz_month": 200_000, "b2b_k": 1_800, "b2c_k": 450, "opex_k": 650, "api_k": 75}

# year 2: 3% рынка; year 3: 8% рынка
Y2_MARKET_SHARE = FIN_Y2["kz_month"] / MARKET_KZ_MONTH
Y3_MARKET_SHARE = FIN_Y3["kz_month"] / MARKET_KZ_MONTH

# Сертификат ГКНТ (571 б.в., базовая величина на 01.06.2026 - 42 BYN)
CERTIFICATE_BV = 571
BASE_VALUE_BYN = 42
CERTIFICATE_BYN = CERTIFICATE_BV * BASE_VALUE_BYN  # 23_982

# ROI Кравира (BYN/мес, осторожные допущения)
ROI_PROTOCOL_COST = KRAVIRA_B2B_MONTH
ROI_METHODIST_FTE_SAVED = 0.35
ROI_METHODIST_FTE_COST = 3_200  # BYN/мес на 1 FTE (loaded)
ROI_METHODIST_SAVING = int(ROI_METHODIST_FTE_SAVED * ROI_METHODIST_FTE_COST)  # 1_120
ROI_CISZ_REJECT_RATE_BASE = 0.04  # 4% пакетов требуют доработки
ROI_CISZ_REJECT_RATE_AFTER = 0.025
ROI_CISZ_REWORK_MINUTES = 25
ROI_CISZ_MINUTE_COST = 2.5  # BYN экв. времени врача/админ
ROI_CISZ_SAVING = int(
    KRAVIRA_KZ_MONTH
    * (ROI_CISZ_REJECT_RATE_BASE - ROI_CISZ_REJECT_RATE_AFTER)
    * ROI_CISZ_REWORK_MINUTES
    * ROI_CISZ_MINUTE_COST
)  # ~2_344
ROI_TOTAL_SAVING = ROI_METHODIST_SAVING + ROI_CISZ_SAVING
ROI_NET = ROI_TOTAL_SAVING - ROI_PROTOCOL_COST


def ebitda_k(year: dict) -> int:
    rev = year["b2b_k"] + year["b2c_k"] + year.get("api_k", 0)
    return rev - year["opex_k"]
