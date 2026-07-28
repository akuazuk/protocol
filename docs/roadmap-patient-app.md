# B2C: «Проверь своё заключение»

Отдельный продукт поверх **Protocol** (B2B: врачи, МИС, методслужба). Пациент получает понятный ответ: насколько заключение согласовано с **клиническими протоколами Минздрава РБ**, что заполнено и что стоит уточнить у врачу - **без** диагноза и **без** силы МЭЭ.

**Статус репозитория:** MVP backend + PWA · июнь 2026 · ветка `main` @ `2aff054`  
**Tier API:** `P1` = L1 structured + deterministic alignment (протоколы через chunk/RAG), **без** ЦИСЗ, send_gate и LLM-критериев  
**Тесты B2C:** `pytest tests/test_patient*.py tests/test_lab_result_parser.py` → **16 passed**

| | B2B (Protocol) | B2C (этот контур) |
|---|----------------|-------------------|
| Кто | Врач, методист, МИС | Пациент, родственник |
| Вопрос | Можно подписать и отправить в ЦИСЗ? | Понятно ли, что врач учёл протокол? |
| Язык | gate_score, alignment, ЦИСЗ | Светофор, блоки, вопросы врачу |
| Данные | FHIR Bundle, полный КЗ | Фото/PDF КЗ, опц. бланки анализов |
| Успех | Меньше возвратов из ЦИСЗ | Пациент пришёл на приём подготовленным |

---

## Зачем это пациенту (Jobs To Be Done)

| Когда… | Пациент хочет… | Продукт даёт… |
|--------|----------------|---------------|
| Только что получил КЗ на руки | Понять «всё ли учли», не читая 5 страниц медицинского текста | Светофор + 2-3 приоритетных темы + «что мы поняли из вашего документа» |
| Есть бланки анализов дома | Убедиться, что врач их учёл в заключении | Сверка маркеров «в анализах / в тексте КЗ» |
| Идёт на повторный приём | Не забыть спросить важное | Чек-лист вопросов + «Поделиться» / печать |
| Сомневается в качестве клиники | Опора на стандарт Минздрава, а не на мнение из чата | Цитаты из клинических протоколов простым языком |

**Главный outcome пилота:** не «оценка в %», а **«я знаю, о чём поговорить с врачом и чувствую контроль»**. Процент - вторичная метрика для прозрачности, не для диагностики.

---

## Аудит текущего MVP (что уже работает)

| Область | Реализовано | Пробел для пользователя |
|---------|-------------|-------------------------|
| Загрузка КЗ + consent | `patient.html`, multipart API | Нет подсказки «сфотографируйте все страницы» / превью страниц |
| Отчёт P1 | `patient_report.py` - светофор, 8 блоков, вопросы | Нет «почему такой статус» (explainability) у каждого блока |
| Анализы | parser + crosscheck + UI panel | OCR фото бланков слабый; мало форматов лабораторий РБ |
| Протокол | `patient_protocol_crosscheck.py` | Показывается только при gap; нет «что протокол рекомендует в целом» |
| Удержание | localStorage: история (5), чек-лист | Нет восстановления полного отчёта после закрытия вкладки |
| Дистрибуция | PWA + SW + manifest | Нет push, нет «Добавить на экран» onboarding на iOS |
| Монетизация | Тарифы в `scripts/konkurs_b2c_ux.py` | Оплата, tier, white-label clinic - не в коде |

---

## Принципы UX (современный стандарт)

Ориентиры: **Apple HIG** (ясность, deference, depth), **Material Design 3** (роли, motion), **NHS Digital Service Manual**, **WCAG 2.2 AA**, паттерны **Ada / K Health / Apple Health** (образование, не диагностика).

### 1. Progressive disclosure - «30 секунд → детали»

```
[Итог одной фразой] → [3 шага что делать] → [Чек-лист врачу] → [Блоки / анализы / протокол - по раскрытию]
```

Первый экран результата **не** таблица из 8 строк. Сначала: статус + plain summary + next steps + CTA «Вопросы на приём».

### 2. Plain language first

- Заголовок блока: «Обследования», не `required_exams`.
- Статус: «Стоит уточнить», «В порядке», «Обратите внимание» - без процентов в заголовке карточки (процент - внутри, по tap).
- Дисклеймер всегда виден, но не пугающий: «Помощник для разговора с врачом».

### 3. Trust & transparency

- «Мы прочитали из вашего документа: …» (`document_read_back_ru`) - выше fold.
- «Сверка с N протоколами Минздрава» + ссылка «Что такое клинический протокол?» (FAQ уже есть - усилить).
- При низком `document_quality` - жёлтый баннер **до** оценки блоков (уже частично есть).

### 4. Accessibility & inclusive design

- [ ] Конtrast ≥ 4.5:1 для текста (проверить `--muted` на `--bg`).
- [ ] Focus ring на drop-zone и кнопках; keyboard-only path.
- [ ] `prefers-reduced-motion` - отключить spin loader / score ring animation.
- [ ] Touch targets ≥ 44×44 px; labels для screen readers на file input.
- [ ] Не полагаться только на цвет (светофор + иконка + текст).

### 5. Mobile-native patterns (PWA сейчас, native позже)

- Document scanner UX: рамка, подсказка освещения, multi-page до отправки.
- Skeleton loaders вместо full-screen overlay на быстрых ответах.
- Web Share API + fallback copy (есть) + **PDF one-pager** «Лист на приём».
- Haptic feedback (Capacitor) на завершении проверки.

### 6. Privacy by design

- Данные не сохраняются на сервере после ответа (явно в UI и privacy notice).
- История только localStorage - объяснить пользователю и предложить «Удалить всё».
- Rate limit + consent checkbox - уже есть; добавить ссылку на политику обработки.

---

## Архитектура модуля (целевое состояние)

### Bounded context «Patient»

```
patient.html (view)
    ↓
/api/patient/*  (rag_server.py - thin controller)
    ↓
clinical_knowledge/patient_review.py  (orchestrator P1)
    ├── consult_tiering.run_l1_structured_review
    ├── patient_lab_crosscheck (optional)
    ├── patient_exams_enrich (optional)
    ├── patient_protocol_crosscheck (optional)
    └── patient_report.build_patient_report
```

**Рекомендации ведущей практики:**

| Практика | Сейчас | Цель |
|----------|--------|------|
| **Contract-first API** | ad-hoc JSON | OpenAPI `/api/patient/openapi.yaml` + JSON Schema `patient_report` v1 |
| **Schema versioning** | неявно | `report_schema_version: 1` в каждом ответе; миграции view |
| **Feature flags** | env vars | `PATIENT_*` + per-clinic flags (`?clinic=`) для white-label |
| **Observability** | логи сервера | structured logs: `patient_review_id`, duration_ms, light, lab_count - **без ПДн** |
| **Idempotency** | нет | `Idempotency-Key` для повторных upload после обрыва сети |
| **SSE progress** | нет | `GET /api/patient/review/stream` - этапы: parse → align → report |
| **Sanitization boundary** | `sanitize_patient_api_payload` | единственная точка; unit-test «B2B fields never leak» |

### Tier-модель (продукт ↔ техника)

| Tier | Продукт | Техника |
|------|---------|---------|
| **P1 Free preview** | Светофор + 3 вопроса (blur остального) | Тот же pipeline, truncate в API |
| **P1 Paid** | Полный отчёт, анализы, протокол | Текущий MVP |
| **P2** | «Объясни простым языком» + evidence pack | L2 + patient-safe narrative + юридический фильтр claims |

---

## Дорожная карта v2 (outcome-driven)

Легенда: `[x]` сделано · `[ ]` запланировано · `[-]` частично

### Волна A - «Полезно с первого дня» (2-3 недели, только PWA)

Фокус: **ценность без App Store и оплаты**.

- [x] Backend P1 + patient_report + crosschecks
- [x] PWA: onboarding, FAQ, чек-лист, share, history
- [x] **Results IA v2:** reorder экрана - summary → next steps → checklist → collapsible blocks/labs/protocol
- [x] **Explainability:** у каждого блока «Почему так» - `why_ru` + excerpt протокола
- [x] **Document quality first:** если confidence < 55% - cap traffic light
- [x] **Restore report:** sessionStorage (`patient-ui.js`)
- [x] **a11y pass:** reduced-motion, focus-visible, aria-live (`patient-tokens.css`)
- [x] **Analytics (privacy-safe):** `POST /api/patient/analytics`

**Критерий готовности:** 5 пользовательских тестов (think-aloud) - все понимают «что делать дальше» без объяснения.

---

### Волна B - «Доверие и дистрибуция» (4-6 недель)

- [x] Privacy notice stub (`docs/patient-privacy-stub.html`)
- [ ] PRD + юридическое заключение (фаза 0)
- [ ] Figma: 7 экранов
- [x] Design tokens: `patient-tokens.css`
- [x] SSE-прогресс: `POST /api/patient/review/stream`
- [x] Оплата stub: `POST /api/patient/payment/session`, `PATIENT_PAYMENT_REQUIRED`
- [x] Query params: `?clinic=&tier=` - white-label + tier bar
- [ ] Soft launch: QR на выдаче КЗ в пилотной клинике

**Критерий:** конверсия QR → завершённая проверка ≥ 15%.

---

### Волна C - «Глубина данных» (6-8 недель)

#### Анализы и обследования

- [x] lab_files, parser, crosscheck UI
- [x] patient_exams_enrich, patient_protocol_crosscheck
- [x] OCR pipeline для фото: `patient_lab_ocr.py` (pytesseract optional)
- [x] Словарь маркеров: расширение Invitro/Synlab/ТТГ и др.
- [ ] Подсказки по типу исследования (ОАК vs УЗИ vs КТ) в patient-friendly формулировках

#### Аккаунт и удержание

- [x] localStorage history + checklist
- [x] Guest session + cloud sync stub: `/api/patient/account/*`
- [x] Local reminder 48h (`btn-set-reminder`)
- [ ] Apple / Google Sign-In → облачная история (encrypted at rest)
- [ ] Push (Capacitor): «Напомнить обсудить с врачом» через 48 ч
- [ ] Абонемент N проверок / месяц

---

### Волна D - «Масштаб B2B2C» (ongoing)

- [x] Партнёрские клиники: white-label config (`patient_clinic_config.py`)
- [ ] SMS/email deep link + rev-share 30/70 (см. `scripts/konkurs_b2c_ux.py`)
- [x] SEO landing `/check` → `patient-check.html`
- [ ] Обезличенная аналитика для Минздрава (агрегаты по gap-темам, без ПДн)
- [x] Tier P2: rule-based narrative (`patient_p2_enrich.py`, tier `detailed`/`onco`)
- [x] Capacitor scaffold: `patient-app/` (App Store path)

---

## Native vs PWA - решение

| Критерий | PWA (сейчас) | Capacitor | React Native |
|----------|--------------|-----------|--------------|
| Time to market | ✅ готово | +2-3 нед | +8-12 нед |
| Camera / OCR | ограничено | ✅ plugins | ✅ |
| IAP / ERIP | сложнее | ✅ | ✅ |
| Offline | SW базовый | ✅ | ✅ |
| Единый код с B2B | ✅ | ✅ | ❌ |

**Рекомендация:** PWA до волны B → **Capacitor 6** для TestFlight, не параллельный RN.

---

## Технический контур (реализовано)

| Комponent | Путь |
|-----------|------|
| Сборка отчёта | `clinical_knowledge/patient_report.py` |
| Оркестрация P1 | `clinical_knowledge/patient_review.py` |
| Парсер анализов | `clinical_knowledge/lab_result_parser.py` |
| Сверка анализов | `clinical_knowledge/patient_lab_crosscheck.py` |
| Блок обследований | `clinical_knowledge/patient_exams_enrich.py` |
| Сверка протокола | `clinical_knowledge/patient_protocol_crosscheck.py` |
| API | `rag_server.py` - `/api/patient/*` |
| PWA | `patient.html`, `patient-manifest.webmanifest`, `patient-sw.js` |
| B2B вход | `index.html` → «Проверь своё заключение» |
| Тесты | `tests/test_patient_report.py`, `test_lab_result_parser.py`, `test_patient_exams_enrich.py`, `test_patient_protocol_crosscheck.py` |
| Экономика B2C | `scripts/konkurs_b2c_ux.py` |

### Tier P1 - pipeline

1. `run_l1_structured_review` - structured 8 блоков + alignment (chunk/RAG по протоколам), без CISZ/LLM criteria.
2. Опционально: lab crosscheck, exams enrich, protocol crosscheck.
3. `build_patient_report` - patient-safe JSON.
4. `sanitize_patient_api_payload` - B2B-поля не отдаются клиенту.

### Переменные окружения

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `PATIENT_REVIEW_ENABLED` | `1` | B2C API |
| `PATIENT_REVIEW_MAX_FILES` | `5` | Макс. файлов КЗ |
| `PATIENT_LAB_MAX_FILES` | `3` | Макс. файлов анализов |
| `RATE_LIMIT_PATIENT_PER_MIN` | `5` | Rate limit по IP |

---

## Тарифы и каналы (продукт)

| Тариф | Цена | Содержание |
|-------|------|------------|
| Промо (клиника) | 2,99 BYN | L1, upsell на tier |
| Базовая L1 | 4,99 BYN | Полный P1 отчёт |
| L1+ (анализы) | 6,99 BYN | P1 + lab + protocol crosscheck |
| L2 подробная | 9,99 BYN | P2 narrative + evidence |
| Онко / pre-op | 12,99-14,99 BYN | Приоритет treatment/safety блоков |

Каналы: QR после визита · SMS rev-share · SEO national (`scripts/konkurs_b2c_ux.py`).

---

## Метрики успеха

| Метрика | Цель (3 мес. пилота) | Как мерить |
|---------|----------------------|------------|
| QR → завершённая проверка | ≥ 15% | UTM + server events |
| Time to «aha» (открыли результат) | < 25 с p95 | RUM |
| Checklist ≥ 1 пункт отмечен | ≥ 30% | localStorage event |
| Share / print | ≥ 20% | click event |
| NPS | ≥ 40 | in-app опрос после share |
| «Задал вопросы врачу» | ≥ 25% | follow-up SMS |
| Жалобы на ошибочную оценку | < 5% | support tag |
| API P1 latency | < 20 с p95 | server metrics |

---

## Тест-стратегия

| Уровень | Что | Статус |
|---------|-----|--------|
| Unit | `patient_report`, parsers, crosschecks | ✅ 16 tests |
| Contract | JSON Schema validation ответа `/api/patient/review` | [ ] |
| Integration | `run_patient_review` end-to-end на фикстурах КЗ | ✅ базовый |
| UI smoke | Playwright: upload → result → share | [ ] |
| a11y | axe-core на `patient.html` | [ ] |
| Load | k6: 10 RPS patient review | [ ] |

Команда перед каждым релизом B2C:

```bash
pytest tests/test_patient*.py tests/test_lab_result_parser.py -q
```

---

## Отличия от B2B Protocol

- Отдельная витрина (`patient.html`), не вкладка врача.
- Упрощённый отчёт: нет Methodist, training, send_gate, ЦИСЗ.
- Отдельный API namespace и rate limits.
- Пациент не видит `gate_score` и сырые alignment cards.

---

## Дисклеймер (UI + API)

Ориентировочная сверка с клиническими протоколами Минздрава РБ. Не является диагнозом, медицинским заключением или заменой очного приёма. При сомнениях обратитесь к лечащему врачу.

`PATIENT_DISCLAIMER_RU` в `patient_report.py`.

---

## Связанные документы

- **Архитектура B2C (для LLM и разработки):** [`docs/architecture-b2c-patient.md`](architecture-b2c-patient.md)
- B2C UX и экономика: `scripts/konkurs_b2c_ux.py`
- Презентация MVP: `docs/mvp-presentation.html#patient-b2c`
- Архитектура L0/L1/L2: `docs/architecture-stages-print.html`
- B2B МИС: `docs/roadmap-mis.md`

---

## Следующий приоритет (июнь 2026)

1. **Волна A** - Results IA v2 + explainability + sessionStorage restore + a11y (максимум пользы без оплаты).
2. **Волна B** - Figma + ERIP + QR пилот в клинике.
3. **Capacitor** - только после стабилизации PWA и конверсии пилота.

*Предыдущая версия roadmap (фазы 0-3) сохранена в истории git; v2 переформулирована в outcome-driven волны A-D.*
