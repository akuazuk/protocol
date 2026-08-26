# Handoff: lab rollout report owner hotfix

Дата: 2026-08-26
Ветка: `hotfix/mo-lab-rollout-report-owner-agent1-pc1`
Worktree: `/private/tmp/protocol-task-mo-lab-rollout-report-owner-pc1`
Base: `a18d7225059c50eb27d90cb81ad659190e889e14`
HEAD implementation commit: `32448355`
PR: https://github.com/akuazuk/protocol/pull/184

Production уже на base SHA. Version и health подтверждены. Rollout state создан,
initial metrics после безопасного operational `chown` записан; guard заблокирован,
primary `0`, PHI-safe health smoke прошёл.

Этот hotfix закрепляет ownership `state/` и `reports/` за cron-пользователем в deploy
и самовосстанавливает write access перед nightly metrics.

Проверки: targeted tests, Ruff, bash syntax и `git diff --check`.
BUILD_VERSION: `2026-08-26-122955Z-mo-lab-report-owner`.

Следующая безопасная команда:
`gh pr checks 184 --repo akuazuk/protocol --watch`

Не трогать параллельно:
`deploy/gcp-app/deploy_to_gce.sh`, `deploy/gcp-app/night_mis_pipeline.sh`.
