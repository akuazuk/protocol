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
| [2026-07-28-multi-machine-git-deploy-workflow-v1.md](2026-07-28-multi-machine-git-deploy-workflow-v1.md) | Workflow для 2 ПК: safe-start, safe-pull, deploy-guard и единый runbook | active |
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
