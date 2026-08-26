# Handoff: лаборатория МО - rollout guard

Дата: 2026-08-26
Репозиторий: `akuazuk/protocol`
Ветка: `cursor/mo-lab-rollout-guard-agent1-pc1`
Worktree: `/private/tmp/protocol-task-mo-lab-rollout-guard-pc1`
Base: `2ae18d4e0c3ac343bffc087f254ecca0fac376c4`
HEAD implementation commit: `c1422ef0`
PR: https://github.com/akuazuk/protocol/pull/182

## Production до этого PR

- PR #179, #180 и #181 merged.
- Primary GCE deploy: SHA `2ae18d4e0c3ac343bffc087f254ecca0fac376c4`.
- BUILD_VERSION: `2026-08-26-113852Z-gce-env-owner`.
- `/health/live` - ok.
- `/api/version` - version и `git_commit` совпали с merge SHA.
- Feature smoke: lab bundle, reconcile и privacy - ok.
- `MO_LAB_BUNDLE=1`, `MO_LAB_SHADOW=1`, `MO_LAB_IN_PRIMARY=0`.
- Первая попытка deploy остановилась до Docker build из-за ownership public env;
  hotfix #181 исправил порядок, повторный deploy успешен.

## В этом PR

- Состояние первого shadow deploy хранится на `/var/data` и не сбрасывается redeploy.
- Ночной PHI-safe отчёт агрегирует lab findings, same-day coverage и решения review pack.
- Primary остаётся неэффективным до 7 успешных ночей и свежего отчёта.
- Требуется минимум 5 решений методиста; false-positive выше 20% блокирует primary.
- Guard применяется и в runtime, и перед GCE deploy.
- Статус и безопасные агрегаты добавлены в `/api/methodist/mo/health`.
- Rollback не меняет warehouse: достаточно `MO_LAB_IN_PRIMARY=0`.

## Проверки

- Полный `pytest -q` - passed, один штатный skip.
- `ruff check .` - passed.
- Shell syntax, Python compile, `git diff --check` - passed.
- Production deploy этого PR не выполнялся.

BUILD_VERSION: `2026-08-26-120249Z-mo-lab-rollout-guard`.

Следующая безопасная команда после публикации:
`gh pr checks 182 --repo akuazuk/protocol --watch`

Не трогать параллельно:
`clinical_knowledge/mo_lab_*.py`, `deploy/gcp-app/night_mis_pipeline.sh`,
`deploy/gcp-app/deploy_to_gce.sh`, `clinical_knowledge/mo_backend.py`.
