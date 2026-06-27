# Навигатор клинических протоколов Минздрава Республики Беларусь

Веб-сервис для **поддержки врача** и **внутреннего контроля** соответствия медицинской документации официальным клиническим протоколам, опубликованным на [сайте Минздрава РБ](https://minzdrav.gov.by/ru/dlya-spetsialistov/standarty-obsledovaniya-i-lecheniya/).

**Не заменяет** очный приём, постановку диагноза, МЭЭ или юридическую экспертизу.

## Для кого

| Роль | Возможности |
|------|-------------|
| **Врач** | Поиск протокола по диагнозу / МКБ-10 / жалобам; развёрнутая выдержка (обследования, лечение, алгоритмы); черновик консультативного заключения |
| **Заведующий / методслужба** | Загрузка PDF консультативного заключения → ориентировочная оценка соответствия протоколам (8 блоков), таблица критериев, зоны риска, отчёт для печати; кабинет методиста (`?mode=methodist`) |
| **Пациент (B2C)** | [`patient.html`](patient.html) - самопроверка выписки по протоколам Минздрава, загрузка фото/PDF, понятный отчёт без полей B2B |
| **Администратор** | Синхронизация корпуса PDF (`download_minzdrav_protocols.py`), оценка качества поиска (`eval/`) |

## Демонстрация (10 минут)

1. Запустите сервер: `pip install -r requirements-rag.txt`, скопируйте `.env.example` → `.env`, укажите ключ API в `.env`, затем `uvicorn rag_server:app --host 127.0.0.1 --port 8787`.
2. Откройте `http://127.0.0.1:8787/?presentation=1` - 4 слайда вступления, демо-сценарии, аналитика пилота, вкладка проверки КЗ первой.
3. Главная страница по умолчанию - **Проверка КЗ** (`#consult-review`); вкладка **Поиск протоколов** - `#search`. Hash сохраняется при обновлении страницы.
4. **Поиск:** демо «I10» / «M32.9 СКВ» или учебный кейс → «Найти протоколы» (быстрее без inline-выдержки; кнопка «Загрузить развёрнутую выдержку»).
5. **Проверка КЗ:** PDF или «Демо-текст КЗ (СКВ)» → «Проанализировать».
6. **B2C для пациентов:** [`patient.html`](patient.html) - отдельная PWA-витрина «Проверь КЗ».
7. Презентация качества MVP: [docs/mvp-presentation.html](docs/mvp-presentation.html) - диаграммы, схема контура качества и дорожная карта (концепт).
8. Буклет: [docs/ministry-brief-print.html](docs/ministry-brief-print.html) → печать в PDF.

## Документы для руководства

- [docs/ministry-brief-ru.md](docs/ministry-brief-ru.md) - краткое описание и KPI пилота
- [docs/mvp-presentation.html](docs/mvp-presentation.html) - презентация MVP для руководства (качество, МИС, Минздрав; концепт-демо графики)
- [docs/deployment-belarus.md](docs/deployment-belarus.md) - развёртывание в контуре РБ
- [docs/roadmap-mis.md](docs/roadmap-mis.md) - этап 2: интеграция с МИС
- [docs/architecture-kravira-fhir-mis-print.html](docs/architecture-kravira-fhir-mis-print.html) - архитектура: КЗ, FHIR BY, ЦИСЗ, МИС «Айболит» (HTML)
- [docs/architecture-kravira-fhir-mis.pdf](docs/architecture-kravira-fhir-mis.pdf) - тот же документ (PDF)
- [docs/pre-sign-checklist-print.html](docs/pre-sign-checklist-print.html) - чек-лист перед подписью ЭЦП для врача (одна страница, печать/PDF)
- [docs/pre-sign-checklist.pdf](docs/pre-sign-checklist.pdf) - тот же чек-лист (PDF)
- [docs/architecture-b2c-patient.md](docs/architecture-b2c-patient.md) - архитектура B2C (`patient.html`, API `/api/patient/*`)
- [docs/project-docs-maintenance.md](docs/project-docs-maintenance.md) - чек-лист синхронизации документов с prod

## Пересчёт метрик качества

```bash
python3 scripts/update_quality_benchmark.py # полный корпус
python3 scripts/update_quality_benchmark.py --mini # smoke
```

## Основные разделы UI

- **Проверка КЗ (PDF)** - главный экран при открытии сайта; прямая ссылка: `#consult-review`. Сверка заключения с фрагментами протоколов; structured analysis по **8 блокам**; сводка для методслужбы и зоны риска (&lt; 70% по критерию). Липкая шапка с минилого Protocol при прокрутке.
- **Поиск протоколов** - `#search`; гибридный отбор (лексика, BM25, семантика), 24 рубрики каталога Минздрава, МКБ-10.
- **Кабинет методиста** - `?mode=methodist` + токен: очередь разметки, ML-дашборд, настройки B2C (`#methodist-queue`, `#ml-dashboard`, `#b2c-monetization`).
- **B2C для пациентов** - [`patient.html`](patient.html): загрузка КЗ, отчёт простым языком, PWA; ссылка также в футере главной страницы.

## API (кратко)

| Метод | Путь | Назначение |
|-------|------|------------|
| POST | `/api/assist` | Подбор протоколов по запросу |
| POST | `/api/protocol-detail` | Развёрнутая выдержка по одному протоколу |
| POST | `/api/protocol-practical` | Выдержка + матрица пунктов для КЗ (≥80% соответствия) |
| POST | `/api/kz-matrix` | Матрица «что должно быть в КЗ» по протоколу |
| POST | `/api/consultation-template` | Черновик консультативного заключения |
| POST | `/api/consult-review` | Проверка 1-3 PDF заключений |
| POST | `/api/icd-suggest` | Подбор кодов МКБ-10 |
| GET | `/api/corpus-stats` | Состояние корпуса (каталог, дата index.csv) |
| GET | `/api/quality-benchmark` | Эталонные метрики подбора |
| GET | `/api/pilot-analytics-demo` | Демо-агрегаты для методслужбы |
| GET | `/api/training-cases` | Учебные клинические кейсы |
| GET | `/api/demo-consult-text` | Текст демо-КЗ (СКВ) |
| POST | `/api/patient/review` | B2C: проверка КЗ для пациента (sanitize, tier) |
| GET | `/api/patient/config` | B2C: конфиг витрины, tiers, monetization |
| GET | `/health` | Готовность RAG и конфигурация |

## Анализ консультативных заключений (КЗ)

Структурный детерминированный разбор КЗ (диагнозы/МКБ, обследования, лекарства с дозами,
красные флаги, применимость протоколов по возрасту/полу/беременности, балл по **8 блокам**)
встроен в проверку КЗ и доступен отдельной командой:

```bash
# один файл (PDF/TXT/JSON)
python -m scripts.analyze_consultation --file path/to/kz.pdf --markdown report.md

# папка целиком (batch; ошибка одного файла не останавливает остальные)
python -m scripts.analyze_consultation --folder data/examples/consultations --output reports/
```

В ответе `/api/consult-review` появляется поле `structured_analysis` (документ + оценка
соответствия) и `report_markdown`. Отключается флагом `CONSULT_STRUCTURED_ANALYSIS=0`.
Подробности: `docs/current_project_audit.md`, `docs/implementation_plan.md`.

## Корпус и актуальность

- Каталог PDF: `download_minzdrav_protocols.py` → `minzdrav_protocols/`.
- Индекс: `index.csv` (~450 протоколов), сверка рубрик: `verify_minzdrav_rubrics.py`.
- Чанки для RAG: `corpus_chunks_parts/` (см. `corpus_chunks_parts/README.md`).

## Качество поиска

См. `eval/README.md`, эталоны `eval/golden_queries.jsonl`, сводка для UI - `data/quality_benchmark.json`.

## Безопасность и ограничения

- Ключ обработки текста хранится на сервере, в браузер не передаётся.
- Консультативные заключения могут содержать персональные данные - используйте защищённый контур и политику ИБ учреждения.
- Оценка соответствия КЗ - **ориентир для методиста**, не вердикт МЭЭ.

## Техническая документация

- Переменные окружения: `.env.example`
- Деплой: `render.yaml`
- Тесты: `tests/`, `pytest`
