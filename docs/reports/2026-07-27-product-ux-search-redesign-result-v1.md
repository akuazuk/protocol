# Отчёт: UX, навигация, поиск и клинические сценарии (задача №2)

**Дата:** 2026-07-27
**Ветка задачи №2:** `codex/product-ux-redesign-v1`
**База (SHA задачи №1):** `c811b04c04962f42a7f51f77e6040c2e3a12cc1d`
**BUILD_VERSION:** `2026-07-27-r17-...` (до) → `2026-07-27-r19-role-navigation-filter-state` (после)

Отчёт по формату §16 ТЗ `docs/plans/2026-07-27-product-ux-search-navigation-redesign-v1.md`.

---

## 1. База и предпосылки

- Задача №1 (`kz-evaluation-quality-v3`) завершена; итоговый SHA `c811b04`.
- На момент финального аудита `main` уже указывает на `c811b04`; задача №2 является
  прямым продолжением этой базы (0 commits behind).
- Создан отдельный clean worktree `/private/tmp/protocol-product-ux-redesign-v1`,
  ветка `codex/product-ux-redesign-v1` от `c811b04`. Основной грязный worktree не тронут.

## 2. Что реализовано (P0/P1/P2)

### P0 - correctness поиска (Workstream A) - ВЫПОЛНЕНО ПОСЛЕ POST-AUDIT FIX

**Новый модуль `clinical_knowledge/search_applicability_gate.py`:**

- Честные статусы результата (§A2): `exact_match`, `icd_match`, `possible`,
  `needs_clarification`, `not_for_audience`, `outdated` + русские подписи.
- Объяснимость (§A3): `why_reasons` (какой код совпал, аудитория документа, условия
  помощи, год, причина уточнения).
- Applicability-gate re-rank (§A1): population/pregnancy/setting/ICD/статус учитываются
  как жёсткие измерения, а не один мягкий score. Неподтверждённое population-specific
  (особенно детское) не может стать Top-1 над нейтральным/подтверждённым.
- «Рекомендуем» только выше порога (§A2): `applicable` + `score>=60` + `exact/icd`.
- Встроено в `clinical_knowledge/protocol_match.py::match_protocol_cards` и в боевой
  быстрый путь `/api/search/protocols-by-icd` за флагом `SEARCH_APPLICABILITY_GATE`
  (default ON), аддитивно - legacy-поля не удаляются.
- Live ICD payload (`path`, `audience`, `matched_icd_codes`, `confidence_score`) приведён
  к контракту gate. При известном клиническом маршруте метка «Рекомендуем» требует
  положительного совпадения темы, а не только наличия сопутствующего кода МКБ.
- Фронтенд сохраняет server-side порядок applicability-gate и больше не пересортировывает
  его по legacy confidence.

**Инвариант приёмки №1 выполнен:** для запроса `I10 эссенциальная гипертензия` без
подтверждённой детской аудитории детский протокол больше не Top-1 и не «Рекомендуем».

### P0 - сломанный расширенный фильтр (§3) - ИСПРАВЛЕНО

Drawer «Дополнительно» получил заголовок «Все фильтры и режим поиска», описание и
`role=region`/`aria-labelledby`. Внутри уже есть рабочие контролы (уровень поиска S0/S1/S2,
режим «только цитаты»); пустой области с одной кнопкой «Закрыть» больше нет.

### P1 - упрощение проверки КЗ (Workstream D) - ВЫПОЛНЕНО

- Вместо выбора L0/L1/L2 - одна галочка «Глубокая сверка с клиническим протоколом»
  (`#consult-deep-check`), синхронизирующая внутренний tier L1/L2.
- Уровни L0/L1/L2 перенесены в раскрываемый `<details id="consult-advanced-tier">`
  (для методиста), значение по умолчанию L1 сохранено.
- Один основной CTA «Проверить заключение»; конкурирующая кнопка «Сверка L2» скрыта
  (`hidden`, обработчик сохранён для совместимости).

### P1/P2 - пациентский сценарий (Workstream E) - ВЫПОЛНЕНО

- Онбординг «Как это работает» свёрнут в `<details id="onboard">` (компактное summary из
  3 шагов) → зона загрузки `#kz-drop` поднялась в первый/начало второго mobile-viewport.
- «Шуточно» убран из основного пути и доступен только за feature flag
  `window.__PATIENT_PLAYFUL_TONE__` (по умолчанию нейтральный «Строго и серьёзно»).

### P1 - design-система и accessibility foundation (Workstream G) - ЧАСТИЧНО

- Новый семантический слой токенов `ux-redesign.css` (`color/space/radius/shadow/font/motion`),
  подключён в `<head>` последним.
- Минимальные размеры (§G2): рабочий текст ≥14px, touch target 44px, видимый `focus-visible`,
  усиленные `border`/`muted` токены для контраста, `prefers-reduced-motion`.

## 3. Изменённые/добавленные файлы

| Файл | Изменение |
|---|---|
| `clinical_knowledge/search_applicability_gate.py` | новый: gate + статусы + объяснимость |
| `clinical_knowledge/protocol_match.py` | wiring gate в `match_protocol_cards` (за флагом) |
| `ux-redesign.css` | новый: токены, a11y, стили простого режима КЗ и статусов |
| `index.html` | ссылка на css; упрощение КЗ; заголовок drawer |
| `patient.html` | онбординг → `<details>`; стили summary |
| `patient-ui.js` | «Шуточно» за feature flag |
| `rag_server.py` | BUILD_VERSION r18 |
| `tests/fixtures/search_applicability_golden.jsonl` | regression dataset (§A4) |
| `tests/test_search_applicability_gate.py` | 15 тестов gate |
| `tests/test_ux_frontend_structure.py` | 11 статических структур/a11y тестов |
| `scripts/bench_search_applicability.py` | before/after benchmark |
| `docs/plans/2026-07-27-product-ux-search-navigation-redesign-v1.md` | статус + прогресс |

## 4. Search benchmark (§13, «Неверная population в Top-1»)

Команда: `python scripts/bench_search_applicability.py`

```
queries considered:               7
BEFORE invalid child Top-1:       4
AFTER  invalid child Top-1:       0   (target 0)
BEFORE recommended child (naive): 5
AFTER  recommended child:         0   (target 0)
RESULT: PASS
```

## 5. Тесты - команды и результаты

```bash
python -m pytest tests/test_search_applicability_gate.py -q      # 15 passed
python -m pytest tests/test_ux_frontend_structure.py -q          # 11 passed
python -m pytest tests/test_api_assist.py tests/test_assist_lite.py \
  tests/test_assist_retrieve_only.py tests/test_search_funnel_api.py \
  tests/test_protocol_search_intents.py tests/test_consult_alignment.py -q  # 22 passed
python -m pytest tests/test_protocol_matcher.py tests/test_protocol_match_venous.py \
  tests/test_protocol_match_ui.py tests/test_protocol_match_detail.py \
  tests/test_search_golden.py tests/test_pediatric_hip_limp_search.py -q     # passed (1 skipped)
python -m ruff check clinical_knowledge/search_applicability_gate.py \
  scripts/bench_search_applicability.py tests/test_search_applicability_gate.py \
  tests/test_ux_frontend_structure.py                             # All checks passed
node --check patient-ui.js && node --check search-flow.js         # OK
```

Пред-существующий lint `F841 icd_roots` в `protocol_match.py:218` - вне диапазона правок
(подтверждено `git diff`), не трогаю по правилу «не менять не относящееся к задаче».

## 6. Таблица метрик «до / после / цель»

| Метрика | До | После | Цель |
|---|---:|---:|---:|
| Неверная population в Top-1 (контрольный набор) | 4 | 0 | 0 |
| Recommended-child при неподтв. аудитории | 5 | 0 | 0 |
| Обязательный выбор L0/L1/L2 в обычном режиме КЗ | да | нет (1 чекбокс) | нет |
| Конкурирующие основные CTA в проверке КЗ | 2 | 1 | 1 |
| «Шуточно» в основном пути пациента | да | за flag | нет |
| Онбординг пациента до зоны загрузки | развёрнут | свёрнут (1 строка) | 1-1.5 экрана |
| Рабочий текст <14px в `ux-redesign.css` | n/a | 0 | 0 |
| Touch target <24×24 в новых контролах | n/a | 0 | 0 |

Метрики Top-1/уточнений/действий и axe/overflow на реальных viewport - см. §7 (блокеры).

## 7. Известные ограничения и блокеры

1. **Полный axe/Lighthouse-аудит ещё не выполнен.** После Cursor выполнен интерактивный
   browser smoke: desktop, mobile 390×844, поиск I10 и `/patient.html`; критический
   сценарий и responsive layout работают. Остаётся прогнать axe-core/Lighthouse и
   расширенную матрицу viewport из §12.
2. Автоматизированная visual-regression матрица пока не добавлена; ключевые desktop-сценарии,
   reload URL-state и auth-screen проверены интерактивно. Структурные/маршрутные регрессии
   закреплены в `tests/test_ux_frontend_structure.py` и `tests/test_workspace_routes.py`.
3. Монолитный `index.html` остаётся техническим долгом. Чистые URL сейчас обслуживают один
   совместимый app shell; вынос экранов в отдельные bundle не требуется для текущей приёмки.

Ни один блокер не ослабляет clinical safety: gate только повышает строгость.

## 8. Ручные проверки после deploy

1. `/` → «Проверить КЗ»: видна одна галочка «Глубокая сверка» и одна кнопка «Проверить
   заключение»; уровни L0/L1/L2 - под «Расширенные настройки».
2. `/` → поиск `I10`: детский протокол не Top-1 и без бейджа «Рекомендуем».
3. `/` → «Дополнительно»: панель с заголовком и рабочими контролами (не пустая).
4. `/patient.html` на 390×844: зона загрузки в первом/начале второго экрана; тон по умолчанию
   нейтральный, «Шуточно» отсутствует.
5. axe-core на ключевых экранах: 0 critical/serious.

## 9. Post-audit: что обнаружено и исправлено

Первичный Cursor-коммит `56581b9` нельзя было безопасно деплоить:

1. applicability-gate был подключён к matcher проверки КЗ, но не к endpoint
   `/api/search/protocols-by-icd`, который реально вызывает врачебный поиск;
2. UI после ответа сервера снова сортировал карточки по `confidence_score`, отменяя
   server-side порядок;
3. первая карточка визуально называлась «Рекомендуем» независимо от поля `recommended`;
4. I10 мог рекомендовать диализ/реабилитацию только из-за сопутствующего кода.

Исправляющий коммит `7e8cf25` закрывает все четыре пункта. Проверка:

```bash
pytest -q tests/test_search_applicability_gate.py tests/test_ux_frontend_structure.py \
  tests/test_api_assist.py tests/test_assist_lite.py tests/test_assist_retrieve_only.py \
  tests/test_search_funnel_api.py tests/test_protocol_search_intents.py \
  tests/test_consult_alignment.py
# 50 passed

ruff check clinical_knowledge/applicability.py \
  clinical_knowledge/search_applicability_gate.py tests/test_search_applicability_gate.py
# All checks passed
```

Browser smoke на 390×844 и desktop подтвердил: профильный протокол гипертензии - Top-1,
только он имеет метку «Рекомендуем», детский протокол не попадает в показанный top-5
для явно взрослой аудитории.

## 10. Git

- Ветка: `codex/product-ux-redesign-v1`
- Cursor commit: `56581b9`
- Post-audit correctness fix: `7e8cf25`
- Push: `git push -u origin codex/product-ux-redesign-v1`

## 11. Завершение отложенных UX-блоков (r19)

После повторного аудита закрыты пункты, оставленные Cursor в §7:

- прямые рабочие URL `/doctor/search`, `/doctor/review`, `/doctor/recent`,
  `/methodist/overview`, `/methodist/cases`, `/methodist/search-quality`; старые hash-ссылки
  остаются совместимыми;
- понятная верхняя навигация «Найти протокол / Проверить КЗ / Пациентам»;
- отдельные фильтры возраста, особого состояния и условий помощи, chips с удалением по одному
  и общим сбросом, восстановление параметров после reload через URL;
- выбранные фильтры больше не очищаются после успешной выдачи;
- однозначный быстрый результат по МКБ скрывает избыточный macro-stepper;
- slug рубрики преобразуется в видимое пользователю название;
- кабинет методиста получил desktop sidebar и защищённый read-only
  `/api/methodist/source-quality` со сводкой аудита корпуса и очередью проверки;
- исправлен root-relative путь `/protocols.json`, обнаруженный browser smoke на вложенных URL.

Проверки r19:

```bash
node --check search-flow.js
pytest -q tests/test_ux_frontend_structure.py tests/test_workspace_routes.py
# 19 passed

pytest -q tests/test_ux_frontend_structure.py tests/test_workspace_routes.py \
  tests/test_health.py tests/test_methodist_feedback.py tests/test_search_funnel_api.py
# 58 passed
```

Интерактивный browser smoke подтвердил прямые URL, выбранные/восстановленные фильтры,
chips, human-readable labels и экран входа методиста. Полный `pytest -q`: 6 baseline
failures, уже зафиксированных и подтверждённых на baseline задачи №1
(`tests/test_assist_search_speed.py` ×2, `test_consult_cache.py`,
`test_drug_normalizer.py` ×2, `test_medication_safety.py`); новых падений от r19 нет.
