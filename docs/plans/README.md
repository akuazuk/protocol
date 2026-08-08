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
| [2026-08-08-mo-action-priority-formula-ru-v1.md](2026-08-08-mo-action-priority-formula-ru-v1.md) | Очередь: приоритет по формуле; P0-P3 → русские слова и цвета; demote stale №55 P0 | active |
| [2026-08-08-mo-icd-first-kp-suggest-v1.md](2026-08-08-mo-icd-first-kp-suggest-v1.md) | КП ICD-first при валидном коде; mismatch только на substantive text; plan 1.0 только с clinical KP | active |
| [2026-08-08-mo-clinical-visit-only-v1.md](2026-08-08-mo-clinical-visit-only-v1.md) | Оценка только clinical_visit; процедуры/стоматология вне score; №55 по пунктам | active |
| [2026-08-08-mo-reg55-day-column-v1.md](2026-08-08-mo-reg55-day-column-v1.md) | Таблица дня: колонка №55 + разбор пунктов при раскрытии | active |
| [2026-08-08-mo-kp-history-episode-suggest-v1.md](2026-08-08-mo-kp-history-episode-suggest-v1.md) | КП по эпизоду Dx из истории визитов + golden верно/неверно | active |
| [2026-08-08-mo-kp-suggest-dx-accuracy-v1.md](2026-08-08-mo-kp-suggest-dx-accuracy-v1.md) | Suggest v4: точный КП по тексту Dx (bridge text→ICD, clinical-only) | archived (преемник: mo-icd-first-kp-suggest-v1; text-path остаётся без кода) |
| [2026-08-07-by-home-gcp-llm-split-v1.md](2026-08-07-by-home-gcp-llm-split-v1.md) | E1: всё на GCP, МИС-мост с Mac → E2 МИС с GCP → E3 BY+LLM на GCP; Docker-границы | active |
| [2026-08-08-mo-icd-dx-matching-pipeline-v3.md](2026-08-08-mo-icd-dx-matching-pipeline-v3.md) | Полный пайплайн Dx↔МКБ: оркестратор, aliases, compact-коды, калибровка→primary, LLM review | active |
| [2026-08-08-mo-patient-history-bundle-v2.md](2026-08-08-mo-patient-history-bundle-v2.md) | Сначала бандл истории пациента (врач + специальность), потом одно МО и анализаторы | active |
| [2026-08-08-mo-icd-name-match-v2.md](2026-08-08-mo-icd-name-match-v2.md) | МКБ name_only: Dx ↔ title_ru; шаги 1-4 в проде | archived (преемник: mo-icd-dx-matching-pipeline-v3) |
| [2026-08-08-mo-prior-dx-usage-baseline-v1.md](2026-08-08-mo-prior-dx-usage-baseline-v1.md) | Prior Dx у врача/специальности (узкий черновик) | archived (преемник: mo-patient-history-bundle-v2) |
| [2026-08-07-mo-dx-text-suggest-icd-directory-eval-v1.md](2026-08-07-mo-dx-text-suggest-icd-directory-eval-v1.md) | КП по тексту диагноза (без МКБ); directory helper v1 | archived (преемник: mo-icd-dx-matching-pipeline-v3; helpers остаются) |
| [2026-08-07-mo-auto-llm-on-disk-v1.md](2026-08-07-mo-auto-llm-on-disk-v1.md) | Night LLM сразу после upload secure_cases на Render (не ждать launchd/merge) | active |
| [2026-08-06-mo-gold-pack-error-sweep-v1.md](2026-08-06-mo-gold-pack-error-sweep-v1.md) | Волна фиксов по gold packs 3650612/3643304: NSAID, suggest, МКБ, re-score, UX | completed (residuals в night handoff 07.08) |
| [2026-08-06-mo-icd-full-document-search-v1.md](2026-08-06-mo-icd-full-document-search-v1.md) | МКБ в оценке МО: искать по всему документу, не только графа «Диагноз» | active |
| [2026-08-06-mo-protocol-suggest-titles-search-v1.md](2026-08-06-mo-protocol-suggest-titles-search-v1.md) | Полные названия КП, поиск из разбора, LLM catch-up 05.08 | active |
| [2026-08-06-mo-case-findings-clarity-v1.md](2026-08-06-mo-case-findings-clarity-v1.md) | Разбор случая: КП 404, ширина разбора, P0 №55 и RU-источники замечаний | active |
| [2026-08-05-mo-case-review-workspace-v2.md](2026-08-05-mo-case-review-workspace-v2.md) | Разбор случая: UI + gold + протоколы МЗ (W0-W3 в проде; gaps в handoff 08-06) | active |
| [2026-08-05-mo-case-review-workspace-v1.md](2026-08-05-mo-case-review-workspace-v1.md) | Разбор случая: sticky МО + scroll разбора, форма решения, RU, таблица дня | archived (преемник: case-review-workspace-v2) |
| [2026-08-05-mo-llm-action-queue-judge-v1.md](2026-08-05-mo-llm-action-queue-judge-v1.md) | LLM-судья A/B только для action-очереди МО (диагноз, затем план) | active |
| [2026-08-05-mo-methodist-review-pack-v1.md](2026-08-05-mo-methodist-review-pack-v1.md) | Отчёты: ID визита/пациента + fullscreen-разбор + пакет решения методиста для обучения | active |
| [2026-08-05-mo-expert-reviewer-portal-v1.md](2026-08-05-mo-expert-reviewer-portal-v1.md) | Кабинет врача-эксперта (логин/пароль), отчёты со вчера, gold из review pack | active |
| [2026-08-05-mo-august-llm-bi-backfill-v1.md](2026-08-05-mo-august-llm-bi-backfill-v1.md) | LLM backfill с августа, continuous на Render, починка merge/BI врачей | active |
| [2026-08-05-mo-eval-smirnova-concordance-v1.md](2026-08-05-mo-eval-smirnova-concordance-v1.md) | Оценка МО: findings согласованности по кейсу Смирнова (статус↔диагноз↔план) | active |
| [2026-08-05-mo-case-protocol-suggest-v1.md](2026-08-05-mo-case-protocol-suggest-v1.md) | Case → Protocol Suggest: подбор КП МЗ по МО/КЗ отдельно от L1 scorer | active |
| [2026-08-04-mo-runtime-stabilization-v1.md](2026-08-04-mo-runtime-stabilization-v1.md) | Стабилизация runtime МО: Docker, вынос пайплайна с Mac; фаза C GCP → [by-home-gcp-llm-split-v1](2026-08-07-by-home-gcp-llm-split-v1.md) | active (C уточнён преемником) |
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
