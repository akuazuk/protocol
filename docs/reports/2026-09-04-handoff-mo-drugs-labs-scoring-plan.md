# Handoff: план МО drugs/labs scoring

Дата: 2026-09-04

| | |
|--|--|
| Repo | `akuazuk/protocol` |
| Branch | `cursor/mo-drugs-labs-scoring-agent1-pc1` |
| Worktree | `/private/tmp/protocol-task-mo-drugs-labs-scoring-pc1` |
| Base | `origin/main` @ `53d61e51` |
| HEAD | `7a81bf1f` |
| PR | https://github.com/akuazuk/protocol/pull/187 |

## Сделано

- План `docs/plans/2026-09-04-mo-drugs-labs-scoring-v1.md`
- Строка в `docs/plans/README.md`
- Docs-only; `BUILD_VERSION` не трогали

## Не сделано

- Код волны 1 (unused lab findings)
- Merge PR

## Следующая команда

После merge #187 - новый worktree на MVP:

```bash
scripts/ops/git_task_start.sh mo-lab-unused-mvp --pc=pc1 \
  --branch=cursor/mo-lab-unused-mvp-agent1-pc1
```

## Не трогать параллельно

Те же зоны, что у lab-from-mis-tests / rceth owners: `mo_lab_*`, `rceth_label_*`, SSOT №55.
