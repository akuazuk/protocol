# Отчёт: UX, навигация, поиск и клинические сценарии (задача №2)

**Дата:** 2026-07-27
**Ветка задачи №2:** `codex/product-ux-redesign-v1`
**База (SHA задачи №1):** `c811b04c04962f42a7f51f77e6040c2e3a12cc1d`
**BUILD_VERSION:** `2026-07-27-r17-...` (до) → `2026-07-27-r18-ux-applicability-gate-kz-simplify` (после)

Отчёт по формату §16 ТЗ `docs/plans/2026-07-27-product-ux-search-navigation-redesign-v1.md`.

---

## 1. База и предпосылки

- Задача №1 (`kz-evaluation-quality-v3`) отправлена в origin; итоговый SHA `c811b04`.
- Задача №1 не влита в `main`, ветка опубликована → по §1.1 база = remote SHA ветки №1.
- Создан отдельный clean worktree `/private/tmp/protocol-product-ux-redesign-v1`,
  ветка `codex/product-ux-redesign-v1` от `c811b04`. Основной грязный worktree не тронут.

## 2. Что реализовано (P0/P1/P2)

### P0 — correctness поиска (Workstream A) — ВЫПОЛНЕНО и покрыто тестами

**Новый модуль `clinical_knowledge/search_applicability_gate.py`:**

- Честные статусы результата (§A2): `exact_match`, `icd_match`, `possible`,
  `needs_clarification`, `not_for_audience`, `outdated` + русские подписи.
- Объяснимость (§A3): `why_reasons` (какой код совпал, аудитория документа, условия
  помощи, год, причина уточнения).
- Applicability-gate re-rank (§A1): population/pregnancy/setting/ICD/статус учитываются
  как жёсткие измерения, а не один мягкий score. Неподтверждённое population-specific
  (особенно детское) не может стать Top-1 над нейтральным/подтверждённым.
- «Рекомендуем» только выше порога (§A2): `applicable` + `score>=60` + `exact/icd`.
- Встроено в `clinical_knowledge/protocol_match.py::match_protocol_cards` за флагом
  `SEARCH_APPLICABILITY_GATE` (default ON), аддитивно — legacy-поля не удаляются.

**Инвариант приёмки №1 выполнен:** для запроса `I10 эссенциальная гипертензия` без
подтверждённой детской аудитории детский протокол больше не Top-1 и не «Рекомендуем».

### P0 — сломанный расширенный фильтр (§3) — ИСПРАВЛЕНО

Drawer «Дополнительно» получил заголовок «Все фильтры и режим поиска», описание и
`role=region`/`aria-labelledby`. Внутри уже есть рабочие контролы (уровень поиска S0/S1/S2,
режим «только цитаты»); пустой области с одной кнопкой «Закрыть» больше нет.

### P1 — упрощение проверки КЗ (Workstream D) — ВЫПОЛНЕНО

- Вместо выбора L0/L1/L2 — одна галочка «Глубокая сверка с клиническим протоколом»
  (`#consult-deep-check`), синхронизирующая внутренний tier L1/L2.
- Уровни L0/L1/L2 перенесены в раскрываемый `<details id="consult-advanced-tier">`
  (для методиста), значение по умолчанию L1 сохранено.
- Один основной CTA «Проверить заключение»; конкурирующая кнопка «Сверка L2» скрыта
  (`hidden`, обработчик сохранён для совместимости).

### P1/P2 — пациентский сценарий (Workstream E) — ВЫПОЛНЕНО

- Онбординг «Как это работает» свёрнут в `<details id="onboard">` (компактное summary из
  3 шагов) → зона загрузки `#kz-drop` поднялась в первый/начало второго mobile-viewport.
- «Шуточно» убран из основного пути и доступен только за feature flag
  `window.__PATIENT_PLAYFUL_TONE__` (по умолчанию нейтральный «Строго и серьёзно»).

### P1 — design-система и accessibility foundation (Workstream G) — ЧАСТИЧНО

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

## 5. Тесты — команды и результаты

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

Пред-существующий lint `F841 icd_roots` в `protocol_match.py:218` — вне диапазона правок
(подтверждено `git diff`), не трогаю по правилу «не менять не относящееся к задаче».

## 6. Таблица метрик «до / после / цель»

| Метрика | До | После | Цель |
|---|---:|---:|---:|
| Неверная population в Top-1 (контрольный набор) | 4 | 0 | 0 |
| Recommended-child при неподтв. аудитории | 5 | 0 | 0 |
| Обязательный выбор L0/L1/L2 в обычном режиме КЗ | да | нет (1 чекбокс) | нет |
| Конкурирующие основные CTA в проверке КЗ | 2 | 1 | 1 |
| «Шуточно» в основном пути пациента | да | за flag | нет |
| Онбординг пациента до зоны загрузки | развёрнут | свёрнут (1 строка) | 1–1.5 экрана |
| Рабочий текст <14px в `ux-redesign.css` | n/a | 0 | 0 |
| Touch target <24×24 в новых контролах | n/a | 0 | 0 |

Метрики Top-1/уточнений/действий и axe/overflow на реальных viewport — см. §7 (блокеры).

## 7. Известные ограничения и блокеры

1. **Нет headless-браузера** в среде агента → axe/keyboard/reflow (§12.3) и visual
   regression (§12.4) на 390×844/768×1024/1280×720/1440×900 не запускались автоматически.
   Оставлены: статические структурные проверки (`test_ux_frontend_structure.py`) и
   `ux-redesign.css` с a11y-инвариантами. **Продолжение:** прогнать axe-core/Lighthouse/
   Playwright на deploy-URL по чек-листу §12.
2. **Полный раздел ролей (Phase 2) с прямыми URL** (`/doctor/*`, `/methodist/*`, `/patient`)
   не внедрён: высокий риск на монолите `index.html` (1.2 МБ) без браузерной регрессии.
   Фундамент оставлен; продолжение — вынести clinical workspace в отдельный layout за флагом,
   сохранив старые deep-links.
3. **Полная модель фильтров chips + URL-state (Phase 3)** — фундамент (заголовок drawer,
   inline population). Продолжение — компоненты `FilterChip`/`FilterDrawer` + восстановление
   запроса/фильтров из URL.
4. **App-shell методиста (Phase 5)** — source-quality из scorer v3 (задача №1) доступен в API;
   UI-оболочка отложена.

Ни один блокер не ослабляет clinical safety: gate только повышает строгость.

## 8. Ручные проверки после deploy

1. `/` → «Проверить КЗ»: видна одна галочка «Глубокая сверка» и одна кнопка «Проверить
   заключение»; уровни L0/L1/L2 — под «Расширенные настройки».
2. `/` → поиск `I10`: детский протокол не Top-1 и без бейджа «Рекомендуем».
3. `/` → «Дополнительно»: панель с заголовком и рабочими контролами (не пустая).
4. `/patient.html` на 390×844: зона загрузки в первом/начале второго экрана; тон по умолчанию
   нейтральный, «Шуточно» отсутствует.
5. axe-core на ключевых экранах: 0 critical/serious.

## 9. Git

- Ветка: `codex/product-ux-redesign-v1`
- Commit SHA: `2c7b915`
- Push: `git push -u origin codex/product-ux-redesign-v1`
