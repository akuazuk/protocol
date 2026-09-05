"""Protocol: дорожная карта развития, платформенные интеграции и health-community (§6.4 БП)."""
from __future__ import annotations

FUTURE_VISION = """
Protocol - самостоятельный продукт и бренд платформы контроля качества амбулаторных заключений.
МЦ «Кравира» - якорный медицинский центр и первый production-пилот (25 000 КЗ/мес, ~1% частного TAM РБ),
но не «название продукта»: Protocol масштабируется на сеть частных ОЗ, пациентов и международные рынки
с единым evidence_map и patient.html.

Дальнейшее развитие строится на трёх слоях: (1) **ядро** - RAG + send_gate + корпус гос. регламентов;
(2) **каналы** - B2B МИС, B2C patient, API/OEM; (3) **платформа** - интеграции с экосистемами
Google/Meta/Apple и community-слой для пациентов с хроническими и онко-диагнозами.
""".strip()

PLATFORM_INTEGRATIONS = """
**Google (Health Connect / Search / Cloud).** После проверки КЗ пациент может сохранить структурированную
выжимку (без сырого PDF) в Health Connect на Android - даты, блоки «что проверено», ссылки на пункты КП.
Для клиник - OAuth2 API «Protocol Gate Status» в Chrome extension для врачей. SEO: rich-results по
запросам «проверить заключение врача [страна]» - органический B2C без CAC. GCP - опциональный хостинг
locale-pack при on-prem by default в РБ/CIS.

**Meta (WhatsApp Business / Messenger).** White-label: клиника отправляет пациенту ссылку на проверку КЗ
через WABA-шаблон (rev-share 30/70). В перспективе 2034+ - **Protocol Care Rooms**: закрытые группы
поддержки при онкологии/ХНИЗ с модерацией методиста; бот отвечает только цитатами из КП (не советы LLM).
Соответствие политикам Meta Health: дисклеймер, без диагноза, human-in-the-loop.

**Apple (Health Records / Wallet).** Экспорт «Protocol Summary Card» в Apple Wallet после B2C-проверки -
QR на отчёт с TTL; интеграция с Health app через FHIR DocumentReference (профиль BY/RU расширяемый).

**Единый Protocol Platform API (2033+).** REST/GraphQL для страховщиков, employer wellness, медтуризма:
«score + evidence_map» без передачи полного текста КЗ; monetization per lookup.
""".strip()

COMMUNITY_VISION = """
**Protocol Community** - мини-соцсеть вокруг доказательной медицины, не «форум болезней».

Функции: (1) анонимные кейсы «мой отчёт vs КП» с redaction ПДн; (2) upvote полезных цитат протокола;
(3) verified badge «проверил через patient.html»; (4) методист-куратор specialty; (5) FAQ по блокам
отчёта (диагноз, лечение, безопасность). Монетизация: freemium + подписка «Хроника Pro» (напоминания,
история проверок, семейный доступ).

Flywheel: community → SEO → B2C → давление на клиники → B2B L0 → больше feedback для ML → лучше отчёты.

Модерация: автоматический фильтр ПДн + методист; запрет медицинских назначений от пользователей;
только цитаты из официального корпуса Protocol.
""".strip()

FUTURE_ROADMAP_TABLE = [
    ("2026-2027", "Protocol РБ", "Production L0/B2C в пилоте МЦ «Кравира»; 1-3 B2B клиента", "MVP"),
    ("2028", "Protocol РБ", "3% TAM B2B; ERIP B2C; API Айболит", "Scale RB"),
    ("2029", "Protocol РБ", "8% TAM; 69,6 тыс. B2C/год; EBITDA +", "Profit RB"),
    ("2030-2031", "Protocol CIS", "KZ, UZ, РФ: corpus + locale RU; Mir/СБП", "Tier 1"),
    ("2031-2032", "Protocol Global EN", "IN, TR: STG/rehber; UPI/locale ML", "Tier 2"),
    ("2032-2033", "Protocol LATAM/EG", "PCDT, EHC CPG; PT/AR packs", "Tier 2+"),
    ("2033-2034", "Protocol Platform", "Google/Meta/Apple connectors; Public API", "Platform"),
    ("2034-2035", "Protocol Community", "Care Rooms; Chronic Pro; insurer API", "Ecosystem"),
    ("2035-2037", "Protocol USA", "Visit Prep; Epic L0; MA payer Verified tier", "US Tier 3"),
]

FUTURE_STREAMS = [
    ("B2B L0/L2 (база)", 1800, "2029"),
    ("B2C patient", 440, "2029"),
    ("Intl B2C upside", 12800, "2033"),
    ("Platform API", 450, "2034"),
    ("Community Pro", 280, "2035"),
    ("Insurer/wellness", 320, "2035"),
    ("USA B2C + B2B + Payer", 53500, "2037"),
]

FUTURE_RISKS = [
    ("Платформенная политика Big Tech", "Средний", "Medical disclaimers, no diagnosis, HITL"),
    ("UGC и ПДн в community", "Высокий", "Redaction, модерация, opt-in pseudonym"),
    ("Регуляторика transborder API", "Средний", "Data residency per country"),
]
