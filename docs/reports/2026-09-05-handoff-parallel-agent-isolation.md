# Handoff: изоляция параллельных агентов и вкладок

Дата: 2026-09-05

## Repo

- repo: `akuazuk/protocol`
- branch: `cursor/parallel-agent-isolation-agent1-pc1`
- worktree: `/private/tmp/protocol-task-parallel-agent-isolation-pc1`
- base: `origin/main` `57d03f20` (#188)
- план: `docs/plans/2026-09-05-parallel-agent-isolation-v1.md`

## Сделано

- CI: `cancel-in-progress: false`, группа `ci-<workflow>-<PR|ref>` - вкладки не гасят
  чужой прогон, новый push той же PR встаёт в очередь.
- `scripts/ops/rebase_task_onto_main.sh` - авто-resolve только `BUILD_VERSION`.
- `scripts/ops/check_pr_file_overlap.sh` - жёсткое пересечение файлов.
- Workflow `PR overlap notify` комментирует, не required, `continue-on-error`.
- AGENTS.md, runbook, cursor-правила.

## Не сделано

- Merge / deploy этого PR.
- Rebase открытого #188 (владелец той вкладки).

## Одна безопасная следующая команда

```bash
# из worktree, после явного «закоммить»:
scripts/ops/bump_build_version.sh parallel-ci-isolation
```
