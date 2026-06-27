# Синхронизация продуктовой документации Protocol

> **Назначение:** чек-лист перед релизом и при крупных изменениях UI/API.  
> **Источник версии:** `BUILD_VERSION` в `rag_server.py` (отдаётся в `/health`, `/api/version`, футере UI).  
> **Автопроверка:** `python3 scripts/check_project_docs.py`

---

## 1. Когда обновлять документы

| Триггер | Что проверить |
|---------|----------------|
| Новая вкладка / hash в `index.html` | `README.md`, `docs/architecture-stages-print.html` §7, презентация |
| Изменения `patient.html` / patient API | `docs/architecture-b2c-patient.md`, бизнес-план §8.1, презентация B2C-блок |
| Режим методиста (`?mode=methodist`) | `docs/methodist-workbench-tz.md`, architecture-stages (не `?methodist=1`) |
| Логотипы / sticky chrome | architecture-b2c §14.1, презентация, буклет |
| Scoring / блоки КЗ (6→8 и т.д.) | audit, implementation_plan, presentation, biz plan |
| FHIR / ЦИСЗ / send_gate | `architecture-kravira-fhir-mis-print.html` + PDF |
| Footer / материалы проекта | `index.html` (4 public + methodist-only), `docs/mvp-presentation.html` footer |

**Правило:** любой осмысленный коммит с изменением кода или user-facing docs → поднять `BUILD_VERSION` и прогнать `check_project_docs.py`.

---

## 2. Карта документов (источник истины)

| Документ | Аудитория | Что должно совпадать с prod |
|----------|-----------|----------------------------|
| `README.md` | разработчики, онбординг | вкладки, hash, B2C, API, демо |
| `docs/mvp-presentation.html` | руководство, инвесторы | 8 блоков, B2C, L0/L1/L2, footer-ссылки |
| `docs/ministry-brief-ru.md` + `ministry-brief-print.html` | Минздрав, дирекция | контуры сервиса, KPI, ИБ |
| `docs/architecture-stages-print.html` (+ PDF) | архитектура по этапам | backend, frontend, methodist, patient |
| `docs/architecture-kravira-fhir-mis-print.html` (+ PDF) | МИС, FHIR, ЦИСЗ | send_gate, cisz, B2C-контур |
| `docs/architecture-b2c-patient.md` | B2C-разработка | API, схемы, бренд, line counts |
| `docs/current_project_audit.md` | compliance KZ | scoring, модули, BUILD_VERSION |
| `docs/konkurs/03_Biznes_plan_*` | конкурс / инвест | B2C MVP status, monetization |
| `.cursor/rules/project-docs-sync.mdc` | агент Cursor | напоминание прогонять чек-лист |

---

## 3. Чек-лист релиза (15 мин)

1. **Версия:** `BUILD_VERSION` в `rag_server.py` - новая дата и `rN`.
2. **Факты из prod:**
   - `curl -s http://127.0.0.1:8787/api/corpus-stats` - число протоколов, chunks (для презентации).
   - Проверить hash-вкладки: `#consult-review`, `#search`, `#methodist-queue`, `#b2c-monetization`.
3. **UI shell:**
   - `index.html`: sticky mini logo + pastel pill tabs в одной строке; footer - 4 карточки для врача (статистика, B2C, презентация, кабинет); буклет, архитектура PDF, чек-лист - только в `?mode=methodist`.
   - `patient.html`: sticky top-bar, hero wordmark.
   - Print-docs: wordmark `protocol-logo-wordmark.svg` в шапке каждого HTML-документа в `docs/`.
4. **Тексты:** нет устаревших «6 блоков», «?methodist=1», «B2C в будущем» (если `patient.html` уже в prod).
5. **Метаданные docs:** в audit / architecture-b2c - строка `Last aligned with code` + текущий `BUILD_VERSION`.
6. **Архитектурный HTML:** если меняли `architecture-*-print.html` → `python3 scripts/build_architecture_pdf.py --all` и коммит PDF.
7. **Автопроверка:** `python3 scripts/check_project_docs.py` - без ошибок (warnings - по ситуации).
8. **Тире UI:** при правках интерфейса - `python3 scripts/normalize_ui_dashes.py`.

---

## 4. Типичные расхождения (ловушки)

| Устарело | Актуально |
|----------|-----------|
| `?methodist=1` | `?mode=methodist` (+ токен методиста) |
| «6 блоков scoring» | **8 блоков** structured compliance |
| «B2C - дорожная карта» | **`patient.html`** PWA в production, API `/api/patient/*` |
| `protocol-logo.svg` = wordmark | emblem (favicon); wordmark = `protocol-logo-wordmark.svg`; mini = `protocol-logo-mini.svg` |
| Вкладка по умолчанию без hash | `#consult-review`, hash сохраняется при F5 |
| «Два контура» без пациента | B2B: поиск + проверка КЗ; B2C: `patient.html` |

---

## 5. Команды

```bash
# Проверка документов
python3 scripts/check_project_docs.py

# Пересборка архитектурных PDF (после правок print HTML)
python3 scripts/build_architecture_pdf.py --all

# Обновление метрик для презентации
python3 scripts/update_quality_benchmark.py --mini
```

---

## 6. История синхронизации

| Дата | BUILD_VERSION | Комментарий |
|------|---------------|-------------|
| 2026-06-24 | `2026-06-24-r40-sticky-logo-docs-sync` | Sticky mini logo в index; чек-лист maintenance; обновление README, stages, b2c, brief, presentation |
| 2026-06-24 | `2026-06-24-r51-footer-methodist-docs-logo` | Footer 4+methodist-only; logo в print-docs; pastel UI в stages; архитектура PDF в ссылках |

*При следующем релизе добавьте строку в эту таблицу.*
