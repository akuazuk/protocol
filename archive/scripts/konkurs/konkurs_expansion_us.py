"""Protocol USA: §6.5 бизнес-плана - pre-visit prep, payer flywheel, страховая экономика."""
from __future__ import annotations

from konkurs_expansion_intl import BYN_PER_USD, intl_b2c_rev_k, intl_checks

# --- Рынок США (оценки CDC/NCHS, CMS, 2024-2025) ---
US_POPULATION_M = 335
US_OFFICE_VISITS_YEAR = 990_000_000  # ambulatory physician office visits
US_ED_VISITS_YEAR = 130_000_000
US_ADDRESSABLE_VISITS = US_OFFICE_VISITS_YEAR  # базовый TAM B2C: амбулаторные визиты PCP/specialist

# Unit economics USA (не в EBITDA 2029)
B2C_US_PRICE_USD = 14.99  # tier Visit Prep / AVS check
B2C_US_PROTOCOL_NET_USD = 10.49  # после rev-share 30% employer/clinic channel
B2C_REV_US_BYN = round(B2C_US_PROTOCOL_NET_USD * BYN_PER_USD, 1)  # ~33,6 BYN Protocol net
B2B_US_L0_USD = 0.35  # per visit gate in EHR (Medicare-scale volume)
PAYER_API_LOOKUP_USD = 0.08  # quality / network tier lookup per member-month active

CONV_US_CAUTIOUS = 0.0001  # 0,01% - SEO + employer pilot
CONV_US_BASE = 0.0005  # 0,05% - payer + Epic channel
CONV_US_UPSIDE = 0.001  # 0,10% - national payer contracts

_us_checks_caut = intl_checks(US_ADDRESSABLE_VISITS, CONV_US_CAUTIOUS)
_us_checks_base = intl_checks(US_ADDRESSABLE_VISITS, CONV_US_BASE)
_us_checks_up = intl_checks(US_ADDRESSABLE_VISITS, CONV_US_UPSIDE)

US_B2C_CAUTIOUS_K = intl_b2c_rev_k(_us_checks_caut, B2C_REV_US_BYN)
US_B2C_BASE_K = intl_b2c_rev_k(_us_checks_base, B2C_REV_US_BYN)
US_B2C_UPSIDE_K = intl_b2c_rev_k(_us_checks_up, B2C_REV_US_BYN)

# B2B upside (2036+): 200 клиник × 8 000 визитов/мес × 12 × $0,35
US_B2B_CLINICS_Y6 = 200
US_B2B_VISITS_MONTH = 8_000
US_B2B_YEAR_USD = US_B2B_CLINICS_Y6 * US_B2B_VISITS_MONTH * 12 * B2B_US_L0_USD
US_B2B_YEAR_K = int(US_B2B_YEAR_USD * BYN_PER_USD / 1000)

# Payer API: 5 млн covered lives × $0,08 × 12
US_PAYER_LIVES_M = 5
US_PAYER_API_YEAR_USD = US_PAYER_LIVES_M * 1_000_000 * PAYER_API_LOOKUP_USD * 12
US_PAYER_API_YEAR_K = int(US_PAYER_API_YEAR_USD * BYN_PER_USD / 1000)

US_ENTRY_COST_K = 420  # HIPAA, legal, corpus, pilot year 1
US_TOTAL_UPSIDE_K = US_B2C_BASE_K + US_B2B_YEAR_K + US_PAYER_API_YEAR_K


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


US_EXPANSION_INTRO = """
**США - приоритетный Tier 3 рынок (§6.5), отдельный от EU/NICE.** Здесь нет единого корпуса «478 КП Минздрава»,
но есть **масштабный амбулаторный поток** (~990 млн office visits/год), доминирование EHR (Epic ~36% больниц),
и **страховая экономика**, где качество визиты напрямую влияет на MLR (Medical Loss Ratio), HEDIS, Star Ratings
Medicare Advantage и премиальные сети.

Protocol в США позиционируется не как «замена UpToDate для врача», а как **Visit Intelligence Platform**:
(1) **Pre-Visit Prep** - пациент готовится к приёму по evidence-based чек-листу;
(2) **After-Visit Summary (AVS) check** - сверка выписки с USPSTF / specialty guidelines / CMS quality measures;
(3) **Payer & Network layer** - страховщик и работодатель стимулируют клиники с Protocol B2B и пациентов с Prep Card.

Кравira-пилот в РБ доказывает ядро (send_gate + evidence_map); US pack - другой корпус и HIPAA, та же архитектура.
""".strip()

US_VALUE_PROP = """
**Проблема США.** Средний PCP-визит - 15-20 мин; пациент приходит неподготовленным, врач тратит время на сбор
анамнеза вместо решения по guideline. ~30% амбулаторных визитов - potentially avoidable (RAND / Health Affairs).
After-Visit Summary часто не соответствует рекомендациям USPSTF или specialty society guidelines. Payer платит за
лишние визиты, imaging и ED returns; клиника теряет HEDIS gaps и Star Rating; пациент - время и copay $25-50.

**Решение Protocol.** До визита: patient.html формирует **Visit Prep Card** - что обсудить, какие скрининги/анализы
по guideline отсутствуют в истории, вопросы врачу (не диагноз). После визита: AVS vs guideline с цитатами.
Клиника с B2B L0 в Epic/Cerner закрывает gaps **до подписи** note. Payer видит агрегированный quality signal без ПДн.
""".strip()

US_INSURANCE_MODEL = """
**Страховая и сетевая экономика (ключевая монетизация США).**

Protocol вводит **двусторонние стимулы** - не штрафы, а рыночные рычаги сети и премии:

**Для пациента (member):**
- **Protocol Prep Card** (employer wellness / payer app): copay снижается с $25-40 до **$5-10** при предъявлении
  Prep Card за 24 ч до визита - payer экономит на avoidable follow-ups и ED.
- **Premium wellness credit**: -$5-15/мес к employee contribution при ≥2 prep visits/год (аналог gym discount).
- Меньше «пустых» визитов: пациент понимает, готов ли он к приёму или лучше telehealth / async message.

**Для клиники / ACO:**
- **Protocol Verified Provider**: клиника с B2B L0 ≥90% ambulatory notes проходит аудит gate → попадание в
  **preferred network tier** payer: +3-8% fee schedule или access к value-based bonus pool.
- Клиника **без** Protocol в сети payer с value-based контрактом: **risk surcharge** на capitation (+1-2%) или
  исключение из «Gold» tier - не наказание регулятора, а **страховой тариф** на более высокий ожидаемый MLR.
- HEDIS gap closure (colorectal screening, diabetes A1c, etc.) - L0 подсказывает до sign-off → Star Rating MA plans.

**Для payer (commercial, MA, self-insured employer):**
- MLR improvement: модель - **-0,8-1,5 pp MLR** на cohort с Prep Card (меньше redundant visits/imaging).
- **Payer API**: $0,06-0,10/member/month за Protocol Network Score клиники (без PHI) - routing пациентов в Verified tier.
- Fraud/waste: AVS check выявляет systematic gaps → не автоматический deny, а **prior auth education** и provider profiling.

**Для работодателя (self-insured):**
- Bundle Protocol в EAP/wellness: $2-4 PEPM (per employee per month) → снижение absenteeism и duplicate specialty visits.

Дисклеймер: Protocol - **CDS / quality transparency**, не utilization management и не замена licensed medical review.
""".strip()

US_STAKEHOLDER_TABLE = [
    (
        "Пациент",
        "Copay $25-50; неструктурированный визит; тревога «правильно ли лечат»",
        "Prep Card −$15-30 copay; чек-лист вопросов; AVS vs guideline; меньше повторных визитов",
        "B2C $9,99-19,99; payer subsidy",
    ),
    (
        "PCP / specialist",
        "15 мин на сбор фактов; пропуск screening gaps",
        "Пациент приходит подготовленным; L0 до sign note; цитаты USPSTF/NCCN в потоке",
        "Epic CDS Hooks; time saved 3-5 мин/visit",
    ),
    (
        "Клиника / health system",
        "HEDIS penalties; MA Star downgrades; malpractice exposure",
        "Protocol Verified tier; quality bonus; 100% note audit L0",
        "B2B $0,25-0,45/visit L0",
    ),
    (
        "Payer (MA / commercial)",
        "MLR 85%+; avoidable utilization; gap in care",
        "Lower MLR; network routing; aggregate quality score",
        "API $0,06-0,10/member/mo",
    ),
    (
        "Self-insured employer",
        "PEPM растёт; sick days",
        "Wellness bundle; fewer duplicate visits",
        "$2-4 PEPM",
    ),
    (
        "ACO / MSSP",
        "Shared savings at risk",
        "Documented guideline adherence pre/post visit",
        "Quality withhold unlock",
    ),
    (
        "CMS / регулятор",
        "Post-hoc audit; patient complaints",
        "Proactive transparency; не замена CMS review",
        "Public health reporting (aggregate)",
    ),
]

US_CORPUS_TABLE = [
    ("USPSTF", "A/B recommendations", "uspreventiveservicestaskforce.org", "Screening gaps"),
    ("CDC / ACIP", "Immunization, infectious", "cdc.gov", "Preventive"),
    ("NCCN", "Oncology pathways", "nccn.org (licensed ingest)", "Onco AVS"),
    ("ACC/AHA", "Cardio guidelines", "professional.heart.org", "CV AVS"),
    ("ADA Standards", "Diabetes care", "diabetes.org", "Chronic"),
    ("CMS Measures", "HEDIS / MIPS / Stars", "cms.gov/qpp", "Quality mapping"),
    ("Epic CDS Hooks", "Local + national rules", "SMART on FHIR", "B2B delivery"),
]

US_PRODUCT_TABLE = [
    ("Visit Prep Card", "B2C / payer", "$9,99-14,99", "24h before PCP/specialist", "Checklist + guideline gaps"),
    ("AVS Check", "B2C", "$14,99-19,99", "After visit", "8-block report vs guidelines"),
    ("L0 Ambulatory Gate", "B2B EHR", "$0,25-0,45/note", "Pre-sign in Epic", "Same as РБ send_gate"),
    ("Payer Network API", "B2P", "$0,06-0,10/member/mo", "Monthly", "Clinic score, no PHI"),
    ("Employer PEPM", "B2B2C", "$2-4/employee/mo", "Annual contract", "Unlimited prep for members"),
]

US_TAM_TABLE = [
    (
        "Осторожная (0,01%)",
        _fmt(US_ADDRESSABLE_VISITS),
        "0,01%",
        _fmt(_us_checks_caut),
        _fmt(US_B2C_CAUTIOUS_K),
        "SEO + 1 employer pilot",
    ),
    (
        "Базовая (0,05%)",
        _fmt(US_ADDRESSABLE_VISITS),
        "0,05%",
        _fmt(_us_checks_base),
        _fmt(US_B2C_BASE_K),
        "Regional payer + Epic pilot",
    ),
    (
        "Upside (0,10%)",
        _fmt(US_ADDRESSABLE_VISITS),
        "0,10%",
        _fmt(_us_checks_up),
        _fmt(US_B2C_UPSIDE_K),
        "National MA + multi-EHR",
    ),
]

US_REVENUE_STACK_TABLE = [
    ("B2C Visit Prep + AVS (базовая conv.)", _fmt(US_B2C_BASE_K), "2035-2036"),
    ("B2B L0 (200 clinics)", _fmt(US_B2B_YEAR_K), "2036-2037"),
    ("Payer API (5M lives)", _fmt(US_PAYER_API_YEAR_K), "2037+"),
    ("Employer PEPM (500k employees)", "38 400", "2036+"),
    ("Итого upside USA (ориентир)", _fmt(US_TOTAL_UPSIDE_K + 38400), "не в EBITDA 2029"),
]

US_GTM_PHASES = [
    (
        "E · 2033",
        "Legal / HIPAA",
        "BAA, SOC2 Type II, US hosting; corpus USPSTF + 20 society GL",
        "Go/no-go",
        str(US_ENTRY_COST_K),
    ),
    (
        "F · 2034",
        "B2C pilot",
        "Visit Prep SEO; 1 000 AVS checks; NPS; no EHR",
        f"B2C path to {_fmt(US_B2C_CAUTIOUS_K)} тыс. BYN",
        "80",
    ),
    (
        "G · 2035",
        "Employer + payer",
        "1 self-insured employer 50k lives; MA regional plan Prep Card copay",
        f"Target {_fmt(US_B2C_BASE_K)} тыс. B2C",
        "120",
    ),
    (
        "H · 2036",
        "Epic SMART",
        "CDS Hooks L0 at 3 health systems; Protocol Verified badge",
        f"+{_fmt(US_B2B_YEAR_K)} тыс. B2B",
        "180",
    ),
    (
        "I · 2037+",
        "Payer network",
        "National API; Gold/Silver network tiers; employer PEPM scale",
        f"Stack ~{_fmt(US_TOTAL_UPSIDE_K)}+ тыс. BYN/год",
        "150/год",
    ),
]

US_GTM_STEPS = [
    (
        "1. Corpus US pack",
        "USPSTF A/B + CMS measures + 20 specialty PDF; map to 8 blocks",
        "8-12 нед.",
        "≥80% top ambulatory conditions",
    ),
    (
        "2. HIPAA & hosting",
        "AWS us-east BAA; zero retention B2C PDF; encryption",
        "2-3 мес.",
        "SOC2 roadmap",
    ),
    (
        "3. Visit Prep MVP",
        "English patient.html; intake questionnaire + record upload",
        "6 нед.",
        "500 beta users",
    ),
    (
        "4. Employer pilot",
        "1 self-insured mid-market; copay incentive $5",
        "3 мес.",
        "Utilization −5% duplicate PCP",
    ),
    (
        "5. Payer copay program",
        "MA plan Prep Card; claims feed aggregate only",
        "6 мес.",
        "10k members enrolled",
    ),
    (
        "6. Epic App Orchard",
        "SMART on FHIR L0; Hyperspace panel",
        "9-12 мес.",
        "1 health system live",
    ),
    (
        "7. Network tier API",
        "Payer pricing actuary sign-off; Verified Provider contract",
        "12-18 мес.",
        "MLR model −1% cohort",
    ),
]

US_INSURANCE_ECON_TABLE = [
    ("Copay пациента (PCP)", "$30", "$8 с Prep Card", "Payer fund −$22 as prevention"),
    ("Visits/patient/year (chronic)", "8,2", "6,9 (−16% avoidable)", "Time + MLR savings"),
    ("HEDIS gap rate (clinic)", "22%", "14% with L0", "Star +0,5 для MA plan"),
    ("Malpractice risk premium", "baseline", "+8% without Verified", "Actuarial network tier"),
    ("Payer MLR (cohort)", "85,2%", "83,8%", "−1,4 pp = $70M/1M lives"),
    ("Clinic fee schedule uplift", "100%", "103-108% Verified", "Quality pool share"),
]

US_RISKS = [
    ("FDA SaMD / CDS guidance", "Средний", "Non-device CDS §520(o)(1)(E) path; no autonomous diagnosis"),
    ("HIPAA / state privacy (CPRA)", "Высокий", "BAA, minimal retention, on-prem B2B option"),
    ("Epic vendor lock-in", "Средний", "SMART universal; Cerner Oracle path parallel"),
    ("Guideline licensing (NCCN)", "Средний", "USPSTF/CMS free; NCCN enterprise license"),
    ("Payer actuarial proof", "Средний", "12-mo cohort study before national tier"),
    ("Anti-kickback / copay subsidy", "Средний", "OIG advisory; fair market copay waivers"),
]

# Для графиков: относительная ценность для стейкholdеров (1-10)
US_STAKEHOLDER_VALUE = [
    ("Пациент", 9),
    ("Payer", 9),
    ("Self-insured employer", 8),
    ("PCP / specialist", 8),
    ("Клиника (Verified)", 8),
    ("ACO / MSSP", 7),
    ("CMS (aggregate)", 6),
]

US_INSURANCE_FLYWHEEL = [
    ("1. Patient Prep", 5),
    ("2. Lower copay incentive", 4),
    ("3. Better visit quality", 5),
    ("4. Clinic joins L0", 5),
    ("5. Payer Verified tier", 4),
    ("6. MLR ↓ premiums stable", 5),
]
