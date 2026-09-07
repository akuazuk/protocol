# Handoff: закрытие remaining-work плана МО

Дата: 2026-09-07.
Repo: `akuazuk/protocol`.
Владелец: agent1 / pc1.
Branch: `cursor/mo-plan-close-agent1-pc1`.
Worktree: `/private/tmp/protocol-task-mo-plan-close-pc1`.
Base: `29b8a1746184527f94e5e66945166d2b6906a2cd`.
PR: открывается этой задачей.

Владелец дал явное разрешение закрыть remaining-work любым способом.
Клинические гейты не выполнены и помечены `owner_deferred`.

## Сверка

- `origin/main` = production `29b8a174`, version `2026-09-07-104339Z-history-same-day`.
- #217-#245 уже merged. Не повторять.
- #246 docs занял старую матрицу - этот отчёт пишет новый файл.
- #113/#186/#194-#203 не мержить.
- Корневой checkout `/Users/pavelkuzauka/Cursor_Folders/Protocol` не менялся.

## Сделано

P1-2. Family KPI:

- `evaluated_cases` и `problem_pct_of_evaluated` по случаям с оценкой;
- ranking только при evaluated N >= 20;
- projection latest CRM `finding_decisions` / статуса на family counts;
- `review.status` = `projected` или `no_reviews`, не `not_projected`.

P1-3:

- E04: HTTP list/detail/export одной synthetic БД, один run/revision/value;
- E20 уже покрыт #243;
- E23: `deploy/gcp-app/verify_lab_assets.py` реально исполняется в pytest.
  Отдельный workflow не добавлялся: `ci.yml` занят dependabot #199.

P1-4:

- Shadow -> «Автоматические сигналы: требуют проверки»;
- поиск выборки vs фильтр строк таблицы;
- абсолютные даты периода и пометка незавершённого периода;
- bulk-бар скрыт при n=0 в analytics и expert;
- column visibility после сортируемых заголовков;
- дубль «Справка» убран из nav.

## Не сделано и снято владельцем

- P1-5 клиническая/нормативная валидация;
- A19 indication graph, A26 case-mix, holdout/split;
- независимый usability walkthrough на проде;
- built-image CI job.

Матрица: `docs/reports/2026-09-07-mo-plan-close-matrix.md`.

## Проверки

```text
pytest tests/test_mo_plan_close_acceptance.py \
       tests/test_mo_meds_labs_dashboards.py \
       tests/test_mo_wave_e_acceptance.py \
       tests/test_mo_review_pack_concurrency.py
```

27 passed. `git diff --check` чистый.

## Файлы этой ветки

- `clinical_knowledge/mo_backend.py`
- `frontend/web/shared/mo-app.js`
- `frontend/web/methodist/mis-kz-quality.html`
- `frontend/web/methodist/expert.html`
- `tests/test_mo_plan_close_acceptance.py`
- `tests/test_mo_meds_labs_dashboards.py`
- `tests/test_mo_wave_e_acceptance.py`
- этот handoff и матрица закрытия
- `rag_server.py` только `BUILD_VERSION`

Не трогать параллельно: dependabot `ci.yml`, #246 матрицу, #113/#186.

## Следующая безопасная команда

После merge этого PR, координатор:

```text
bash deploy/gcp-app/deploy_to_gce.sh
curl -fsS https://protocol.kravira.by/health/live
curl -fsS https://protocol.kravira.by/api/version
```

Не деплоить, пока HEAD не равен `origin/main`. Render не прод.
