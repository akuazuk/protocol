# Планы работ Protocol (источник истины по задачам)

Здесь лежат **версионированные планы** работ. Перед началом любой осмысленной задачи
агент/разработчик **обязан** прочитать актуальный план из этого каталога и, при
необходимости, создать новую версию.

## Правила ведения

1. Один файл - один план, имя: `YYYY-MM-DD-<тема>-vN.md` (латиница, kebab-case).
2. Новая итерация плана - **новый файл** с `vN+1`, старую версию не редактируем задним числом,
   а помечаем в этом индексе как `archived` и указываем преемника.
3. Каждый план содержит: контекст, что изменено в проде, метрики (было / стало / цель),
   шаги (сделано / в работе / дальше), риски.
4. Перед коммитом задачи - обнови статус шагов и метрик в актуальном плане.
5. Не плоди мусор: черновики и разовые заметки сюда не кладём.

## Индекс планов

| Файл | Тема | Статус |
|------|------|--------|
| [2026-08-05-mo-llm-action-queue-judge-v1.md](2026-08-05-mo-llm-action-queue-judge-v1.md) | LLM-судья A/B только для action-очереди МО (диагноз, затем план) | active |
| [2026-08-05-mo-methodist-review-pack-v1.md](2026-08-05-mo-methodist-review-pack-v1.md) | Отчёты: ID визита/пациента + fullscreen-разбор + пакет решения методиста для обучения | active |
| [2026-08-05-mo-expert-reviewer-portal-v1.md](2026-08-05-mo-expert-reviewer-portal-v1.md) | Кабинет врача-эксперта (логин/пароль), отчёты со вчера, gold из review pack | active |
| [2026-08-05-mo-eval-smirnova-concordance-v1.md](2026-08-05-mo-eval-smirnova-concordance-v1.md) | Оценка МО: findings согласованности по кейсу Смирнова (статус↔диагноз↔план) | active |
| [2026-08-05-mo-case-protocol-suggest-v1.md](2026-08-05-mo-case-protocol-suggest-v1.md) | Case → Protocol Suggest: подбор КП МЗ по МО/КЗ отдельно от L1 scorer | active |
| [2026-08-04-mo-runtime-stabilization-v1.md](2026-08-04-mo-runtime-stabilization-v1.md) | Стабилизация runtime МО: Docker, вынос пайплайна с Mac, опционально GCP | active |
| [2026-08-04-repo-sections-archive-v2.md](2026-08-04-repo-sections-archive-v2.md) | Карта разделов продукта и безопасная архивация (konkurs, ML dumps, hygiene) | active |
| [2026-08-03-ci-release-concurrency-v3.md](2026-08-03-ci-release-concurrency-v3.md) | CI baseline 0, обязательный lint и GitHub Actions concurrency для production release | completed |
| [2026-08-03-mo-filter-actions-ui-v1.md](2026-08-03-mo-filter-actions-ui-v1.md) | МО Аналитика: явный запуск поиска, подтверждение мультифильтров и раскрываемые панели | active |
| [2026-08-03-multi-agent-release-guard-v2.md](2026-08-03-multi-agent-release-guard-v2.md) | Multi-agent release guard: только merged `origin/main` в Render, hard guards и branch protection | archived (преемник: ci-release-concurrency-v3) |
| [2026-08-03-mo-rubric-mz-scoring-viz-v1.md](2026-08-03-mo-rubric-mz-scoring-viz-v1.md) | МО: рубрика МЗ «Как оценивать» (0/0.5/1), shadow scorer и визуализация в case detail | active |
| [2026-07-30-mo-analytics-bi-redesign-v1.md](2026-07-30-mo-analytics-bi-redesign-v1.md) | МО Аналитика: редизайн BI (ECharts, фильтры, «Вчера» и «Месяц»), доставка данных в прод и оценка v4 | active |
| [2026-07-29-repo-structure-cleanup-v1.md](2026-07-29-repo-structure-cleanup-v1.md) | Cleanup структуры репозитория: frontend/backend/ops без регрессий прода | archived (преемник: repo-sections-archive-v2) |
| [2026-07-28-multi-machine-git-deploy-workflow-v1.md](2026-07-28-multi-machine-git-deploy-workflow-v1.md) | Workflow для 2 ПК: safe-start, safe-pull, deploy-guard и единый runbook | archived (преемник: multi-agent-release-guard-v2) |
| [2026-07-28-mo-daily-bi-platform-v1.md](2026-07-28-mo-daily-bi-platform-v1.md) | МО из БД: ежедневная загрузка «вчера», объективная оценка, CRM/BI, отчёты, терминология КЗ/МО и полный перенесённый backlog | active |
| [2026-07-27-product-ux-search-navigation-redesign-v1.md](2026-07-27-product-ux-search-navigation-redesign-v1.md) | UX, applicability-gate, роли, фильтры и навигация | archived (реализовано до r20; незавершённое перенесено в mo-daily-bi-platform-v1) |
| [2026-07-27-kz-evaluation-quality-overnight-v1.md](2026-07-27-kz-evaluation-quality-overnight-v1.md) | Scorer v3, trust, coverage/confidence, knowledge-model, gold-инфраструктура | archived (P0 реализован в r17; P1/P2 перенесены в mo-daily-bi-platform-v1) |
| [2026-07-27-mis-kz-dashboard-rubric-v1.md](2026-07-27-mis-kz-dashboard-rubric-v1.md) | MIS · КЗ: единая рубрика дашборда и фильтры | archived (реализовано до r20; преемник: mo-daily-bi-platform-v1) |
| [2026-07-22-kz-deep-eval-db-task-v1.md](2026-07-22-kz-deep-eval-db-task-v1.md) | Глубокая оценка данных МИС и клинические проверки | archived (преемник: mo-daily-bi-platform-v1) |
| [2026-07-22-kz-data-separation-viz-v1.md](2026-07-22-kz-data-separation-viz-v1.md) | Разделение клинических и не-клинических записей | archived (преемник: mo-daily-bi-platform-v1) |
| [2026-07-22-kz-scoring-methodology-v1.md](2026-07-22-kz-scoring-methodology-v1.md) | Методология оценки и обогащение протоколов | archived (преемник: scorer v3 в mo-daily-bi-platform-v1) |
| [2026-07-22-mis-kz-pay-services-l1-v1.md](2026-07-22-mis-kz-pay-services-l1-v1.md) | Данные МИС, оплата, услуги и L1 | archived (преемник: mo-daily-bi-platform-v1) |
| [2026-07-22-mis-kz-quality-analysis.md](2026-07-22-mis-kz-quality-analysis.md) | Исторический анализ качества L1 после догрузки | archived (исторический отчёт; преемник: mo-daily-bi-platform-v1) |
| [2026-07-21-mis-kz-llm-progress-full-report-v1.md](2026-07-21-mis-kz-llm-progress-full-report-v1.md) | Прогресс LLM-прогона + полный разбор по протоколам МЗ | archived (преемник: mis-kz-pay-services-l1-v1) |
| [2026-07-21-mis-kz-worst50-l2-gemini-v1.md](2026-07-21-mis-kz-worst50-l2-gemini-v1.md) | Топ-50 худших КЗ + L2 + выборочный Gemini | archived (преемник: llm-progress-full-report-v1) |
| [2026-07-21-mis-kz-l1-batch-v1.md](2026-07-21-mis-kz-l1-batch-v1.md) | Массовый L1 mis_protocol (июль: 7648 визитов) + дашборд методиста | archived (преемник: worst50-l2-gemini-v1) |
| [2026-07-20-protocol-reextract-quality-v1.md](2026-07-20-protocol-reextract-quality-v1.md) | Чистка навигатора протоколов + LLM-переизвлечение 373 auto-протоколов | archived (преемник: работа закрыта в r23; дальше - mis-kz-l1) |

## Легенда статусов

- `active` - текущий план, ему следуем.
- `blocked` - ждём внешнего события (доступ, кредиты API, решение владельца).
- `done` - выполнен, оставлен для истории.
- `archived` - устарел, см. столбец преемника.
