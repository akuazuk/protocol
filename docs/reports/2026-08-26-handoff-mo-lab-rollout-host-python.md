# Handoff: lab rollout host Python hotfix

Дата: 2026-08-26
Репозиторий: `akuazuk/protocol`
Ветка: `hotfix/mo-lab-rollout-host-python-agent1-pc1`
Worktree: `/private/tmp/protocol-task-mo-lab-rollout-host-python-pc1`
Base: `085ff7d6f212f855f1e79947c88c0356193fd245`
PR: будет создан после push.

## Причина

Deploy PR #182 остановился до Docker build/run: host Python не содержит `pydantic`,
а metrics runner импортировал `clinical_knowledge/__init__.py`. Production остался
на SHA `2ae18d4e0c3ac343bffc087f254ecca0fac376c4`.

## Исправлено

- Metrics runner загружает stdlib-only модуль напрямую и работает с `python -S`.
- Env assembler считает ключи до передачи файла cron-владельцу и не пишет ложный
  permission warning после `chmod 600`.

## Проверки

- Rollout/release tests - passed.
- Ruff, bash syntax, `git diff --check` - passed.
- Production deploy этого hotfix ещё не выполнялся.

BUILD_VERSION: `2026-08-26-121434Z-mo-lab-host-python`.

Следующая безопасная команда после публикации:
`gh pr checks <PR> --repo akuazuk/protocol --watch`

Не трогать параллельно:
`scripts/run_mo_lab_rollout_metrics.py`,
`deploy/gcp-app/assemble_web_env_from_sm.sh`.
