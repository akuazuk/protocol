# Handoff: лаборатория МО production-ready

Дата: 2026-08-26
Репозиторий: `akuazuk/protocol`
Ветка: `cursor/mo-lab-production-ready-agent1-pc1`
Worktree: `/private/tmp/protocol-task-mo-lab-production-ready-pc1`
Base: `9e4e5322` (`origin/main`)
HEAD: будет зафиксирован после commit
PR: будет создан после push; заменяет PR #178

## Сделано

- Перенесён итоговый diff лаборатории в чистую ветку от свежего `origin/main`.
- Reconcile отделён от обрезанного UI payload и использует полный безопасный
  panel/indicator index без значений и identity.
- `indicator_name` участвует в распознавании панели; указание родительской панели
  не создаёт ложный gap по вложенному показателю.
- Результаты после даты визита видны как контекст, но не создают finding.
- Live и batch evaluation используют один путь с одинаковым primary/shadow поведением.
- Ingest заменяет диапазон после полного чтения МИС, удаляет точные дубли и создаёт
  unique-индекс.
- Nightly append остаётся non-fatal для основного МО, но получил отдельный status,
  retry и Telegram alert через штатный checker.

## Тесты

- `pytest -q` - полный suite прошёл, один штатный skip.
- Узкий lab suite - 31 passed.
- `ruff check .` - passed.
- `python3 -m py_compile` изменённых Python runtime-файлов - ok.
- `node --check frontend/web/shared/mo-app.js` - ok.
- `bash -n` nightly/check scripts - ok.
- `git diff --check HEAD` - выполнить повторно перед commit.

## Не сделано

- Merge и primary GCE deploy не выполнялись.
- `MO_LAB_IN_PRIMARY=1` в GCE env не включён.
- Числовые значения не влияют на score без референсов и калибровки.

BUILD_VERSION: `2026-08-26-104633Z-mo-lab-ready`.
Production SHA / smoke: отсутствуют, deploy не выполнялся.

Следующая безопасная команда после публикации:
`gh pr checks --watch <replacement-pr>`

Не трогать параллельно:
`clinical_knowledge/mo_lab_*.py`, `scripts/ingest_mo_lab_from_mis_tests.py`,
`deploy/gcp-app/night_mis_pipeline.sh`, `frontend/web/shared/mo-app.js`.
