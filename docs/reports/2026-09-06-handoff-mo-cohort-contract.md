# Handoff: МО — календарный срез API, этап 1a

Дата: 2026-09-06. Repo: akuazuk/protocol. Владелец: agent1 / pc1.
Branch: `codex/mo-cohort-contract-agent1-pc1`.
Worktree: `/private/tmp/protocol-task-mo-cohort-contract-pc1`.
Base: `a592d588fdd7eb428161024ad13e4e3948bb3754`.
HEAD: commit с этим handoff; точный опубликованный SHA и PR указаны в GitHub.

## Сверка Cursor и production перед работой

В main вошли #148 (billed key для history deep), #158 (MIS только GCE), #204
(requirements-rag.lock в deploy allowlist), после #192/#193. Production GET
`/api/version` подтвердил `a592d588fdd7eb428161024ad13e4e3948bb3754`, version
`2026-09-06-073651Z-deploy-lock-allowlist`; `/health/live` ok, rag_ready true.
Read-only docker inspect: `protocol-gcp-app:a592d588fdd7`, running,
started_at `2026-09-06T08:44:05.996563403Z`, restart_count=0. Buildkit workers
на момент проверки не обнаружены; это моментальный снимок, не блокировка будущих релизов.
Исходный грязный checkout Cursor не изменялся.

## Изменение

API cases/facets/overview/drugs-labs-kpis принимает period/month. Legacy record
paths нормализуют современные календарные параметры через общий resolve_periods.
Drug/lab SQL использует общий _sql_case_filter (период, филиал, врач,
специальность, статус, тип документа) и пересекает тип с clinical eligibility.
Ответ возвращает фактические границы. Неверный календарный запрос даёт 422.
Исторический month, yesterday и 7d больше не теряются между UI и API.

Это этап 1a, не закрытие всего roadmap. Остаются расширенные фильтры/№55,
семантика знаменателей lab, unknown/family dedupe, history episode, lab wiring,
лекарственные правила, evidence и клиническая валидация. Primary flags не менялись.

## Проверки

53 passed: test_mo_cohort_contract, test_mo_meds_labs_dashboards,
test_mo_backend, test_mo_month, test_mo_metrics. HTTP→SQLite parity на полностью
синтетическом складе: соседние месяцы, текущий неполный месяц, вчера/7д/custom,
филиал/статус/врач, типы, пустой период, неверный period, старые custom-даты.
Ruff изменённых файлов, py_compile, git diff --check пройдены.
Полный CI должен быть проверен на опубликованном HEAD до merge.

BUILD_VERSION: `2026-09-06-084802Z-mo-cohort-contract`.
На момент первого commit: merge/deploy этого изменения не выполнены.
Schema/data migrations отсутствуют. После релиза сверить version/SHA и API
на одном месяце/типе/статусе; не печатать клинические записи/ID/токены.

## Координация

Держим `clinical_knowledge/mo_backend.py`, route-блоки cases/facets/overview/
drugs-labs-kpis в `rag_server.py`, `tests/test_mo_cohort_contract.py` и этот handoff.
#186/#113 затрагивают rag_server.py только BUILD_VERSION: проверены hunks,
это soft overlap по pr_isolation.classify_overlap(other_rag_only_version=True).
Старый shell overlap wrapper не определяет version-only peer и может назвать
его HARD: при повторе проверять фактический diff, не менять защитные скрипты
попутно. Другая вкладка должна rebase после merge; не заменять наши маршруты.
Новые тесты — регрессия данного модуля; барьеры CI/порог покрытия не менялись.

Безопасная следующая команда:

```bash
gh pr list --repo akuazuk/protocol --state open
```
