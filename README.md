# Навигатор клинических протоколов Минздрава Республики Беларусь

Веб-сервис для **поддержки врача** и **внутреннего контроля** соответствия медицинской документации официальным клиническим протоколам, опубликованным на [сайте Минздрава РБ](https://minzdrav.gov.by/ru/dlya-spetsialistov/standarty-obsledovaniya-i-lecheniya/).

**Не заменяет** очный приём, постановку диагноза, МЭЭ или юридическую экспертизу.

## Для кого

| Роль | Возможности |
|------|-------------|
| **Врач** | Поиск протокола по диагнозу / МКБ-10 / жалобам; развёрнутая выдержка (обследования, лечение, алгоритмы); черновик консультативного заключения |
| **Заведующий / методслужба** | Анализ отдельного документа (КЗ или медосмотр), аналитика МО из БД, очередь разбора, динамика по врачам/специальностям и отчёты |
| **Пациент (B2C)** | [`patient.html`](patient.html) - самопроверка выписки по протоколам Минздрава, загрузка фото/PDF, понятный отчёт без полей B2B |
| **Администратор** | Синхронизация корпуса PDF (`download_minzdrav_protocols.py`), оценка качества поиска (`eval/`) |

## Демонстрация (10 минут)

1. Запустите сервер: `pip install -r requirements-rag.txt`, скопируйте `.env.example` → `.env`, укажите ключ API в `.env`, затем `uvicorn rag_server:app --host 127.0.0.1 --port 8787`.
2. Откройте `http://127.0.0.1:8787/?presentation=1` - 4 слайда вступления, демо-сценарии и аналитика пилота.
3. Главная рабочая вкладка - **Анализ документа** (`#consult-review`): принимает КЗ или медосмотр. Вкладка **Поиск протоколов** - `#search`.
4. **Поиск:** демо «I10» / «M32.9 СКВ» или учебный кейс → «Найти протоколы» (быстрее без inline-выдержки; кнопка «Загрузить развёрнутую выдержку»).
5. **Анализ документа:** PDF или демо-текст -> «Проанализировать».
6. **B2C для пациентов:** [`patient.html`](patient.html) - отдельная PWA-витрина «Проверь КЗ».
7. Презентация качества MVP: [docs/mvp-presentation.html](docs/mvp-presentation.html) - диаграммы, схема контура качества и дорожная карта (концепт).
8. Буклет: [docs/ministry-brief-print.html](docs/ministry-brief-print.html) → печать в PDF.

## Документы для руководства

- [docs/ministry-brief-ru.md](docs/ministry-brief-ru.md) - краткое описание и KPI пилота
- [docs/mvp-presentation.html](docs/mvp-presentation.html) - презентация MVP для руководства (качество, МИС, Минздрав; концепт-демо графики)
- [docs/deployment-belarus.md](docs/deployment-belarus.md) - развёртывание в контуре РБ
- [docs/roadmap-mis.md](docs/roadmap-mis.md) - этап 2: интеграция с МИС
- [docs/architecture-kravira-fhir-mis.pdf](docs/architecture-kravira-fhir-mis.pdf) - архитектура: КЗ, FHIR BY, ЦИСЗ, МИС «Айболит» (PDF; HTML-исходник для печати: `architecture-kravira-fhir-mis-print.html`)
- [docs/pre-sign-checklist-print.html](docs/pre-sign-checklist-print.html) - чек-лист перед подписью ЭЦП для врача (одна страница, печать/PDF)
- [docs/pre-sign-checklist.pdf](docs/pre-sign-checklist.pdf) - тот же чек-лист (PDF)
- [docs/architecture-b2c-patient.md](docs/architecture-b2c-patient.md) - архитектура B2C (`patient.html`, API `/api/patient/*`)
- [docs/project-docs-maintenance.md](docs/project-docs-maintenance.md) - чек-лист синхронизации документов с prod

## Пересчёт метрик качества

```bash
python3 scripts/update_quality_benchmark.py # полный корпус
python3 scripts/update_quality_benchmark.py --mini # smoke
```

## Структура репозитория (канонично)

- `backend/` - backend entrypoint и серверный контур.
- `frontend/` - целевой дом для web/patient интерфейсов (миграция по фазам).
- `clinical_knowledge/` - доменная и клиническая логика.
- `scripts/` - утилиты сборки, batch, deploy, ops.
- `docs/` - архитектура, планы, отчёты, runbook.
- `data/`, `output/`, `ml/` - данные/артефакты/эксперименты.
- `tests/` - unit/integration/regression тесты.

План cleanup структуры: `docs/plans/2026-07-29-repo-structure-cleanup-v1.md`.

## Работа с двух компьютеров (3 команды)

Чтобы не запутаться с ветками, pull и deploy, используйте только этот сценарий:

```bash
# 1) старт сессии + подсказки по состоянию репо
scripts/ops/git_safe_start.sh

# 2) новая задача: авто-ветка + clean worktree (подставьте slug и pc1/pc2)
scripts/ops/git_task_start.sh <task-slug> --pc=pc1

# 3) перед pull и deploy - обязательные guard-проверки
scripts/ops/git_safe_pull.sh
scripts/ops/git_deploy_guard.sh --prod-url=https://protocol-bimy.onrender.com

# если deploy идёт в Render напрямую из Git (ветка подключения, обычно main)
scripts/ops/git_deploy_guard.sh --render-git --render-branch=main --prod-url=https://protocol-bimy.onrender.com

# one-shot wrapper: push main + guard
scripts/ops/deploy_after_push.sh --branch=main --prod-url=https://protocol-bimy.onrender.com

# one-shot wrapper: push + guard + ожидание целевой версии на Render
scripts/ops/deploy_after_push.sh --branch=main --prod-url=https://protocol-bimy.onrender.com --wait-version
```

Подробный runbook: `docs/deploy/multi-machine-git-deploy-runbook.md`.

Hygiene-audit рабочей копии (read-only): `scripts/ops/check_repo_hygiene.sh`.

Совместимость: старые пути `scripts/*.sh` пока сохранены и продолжают работать.

## Основные разделы UI

- **Анализ документа** - `#consult-review`; одиночная загрузка КЗ или медосмотра, автоматическое определение типа и проверка по клиническим протоколам.
- **Поиск протоколов** - `#search`; гибридный отбор (лексика, BM25, семантика), 24 рубрики каталога Минздрава, МКБ-10.
- **МО · аналитика** - массовая оценка данных из БД МИС: отчёт за вчера, динамика, врачи, специальности и очередь разбора.
- **Кабинет методиста** - `?mode=methodist` + токен: очередь разметки, ML-аналитика и настройки пациентского сервиса.
- **B2C для пациентов** - [`patient.html`](patient.html): загрузка КЗ, отчёт простым языком, PWA; ссылка в футере главной страницы.
- **Статистика поиска** - `#analytics-stats`; кнопка в футере (корпус, качество подбора, метрики КЗ).

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
| GET | `/api/methodist/mo/overview` | Сводка массовой аналитики МО из БД |
| GET | `/api/methodist/mo/daily-report` | Ежедневный отчёт МО |
| GET | `/health` | Готовность поиска и конфигурация |

## Одиночный анализ медицинского документа

Структурный машинный разбор КЗ или медосмотра (диагнозы/МКБ, обследования, лекарства с дозами,
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
Подробности: `docs/current_project_audit.md`.

## Корпус и актуальность

- Каталог PDF: `download_minzdrav_protocols.py` → `minzdrav_protocols/`.
- Индекс: `index.csv` (~450 протоколов), сверка рубрик: `verify_minzdrav_rubrics.py`.
- Фрагменты для поиска по протоколам: `corpus_chunks_parts/` (см. `corpus_chunks_parts/README.md`).

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
