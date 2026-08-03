# Handoff: МО Аналитика, Git-ветки и multi-agent workflow

Дата: 2026-08-03
Repository: `akuazuk/protocol`
Production: `protocol` / `srv-d78he6h5pdvs73b1kufg`

## 1. Что сделано и находится в production

Merged PR:

1. PR #1, merge SHA `b1a0b45b` - warehouse detail и исходное МО по клику.
2. PR #2, merge SHA `b663020a` - навигация, breadcrumbs, health/capabilities,
   доступные таблицы, secure source fallback, удаление синей рамки рабочей области и
   Cursor-план улучшений.
3. PR #3, merge SHA `e960b9ff` - чтение secure CSV без runtime-зависимости от pandas.

Production smoke после PR #3:

- `/api/methodist/mo/capabilities` отвечает `200`;
- warehouse `ready`, freshness `fresh`, pipeline `success`;
- очередь и detail содержат скоры;
- source `secure_csv`, state `ready`;
- жалобы и анамнез присутствуют;
- ID и clinical text в диагностические логи не выводились.

Health остаётся `degraded` только по `report_reconciliation_mismatch`.

## 2. Почему production продолжает показывать r12

В PR #2 и #3 не был обновлён `BUILD_VERSION` в `rag_server.py`, поэтому `/api/version`
продолжает возвращать `2026-08-03-r12-mo-warehouse-case-detail`, хотя новый код реально
развёрнут. `/api/version` сейчас не выводит Git SHA.

В ветке `codex/multi-agent-repo-runbook` уже подготовлена версия
`2026-08-03-r13-multi-agent-runbook`. После её PR merge/deploy нужно:

1. проверить, что production показывает `r13`;
2. отдельной backend-задачей добавить Git SHA/deploy time в `/api/version`;
3. повторить version + feature smoke.

## 3. Аудит веток на момент handoff

`origin/main = e960b9ff`. Открытых PR нет.

| Ветка | Состояние | Решение |
|---|---|---|
| `origin/codex/mo-analytics-data-flow-v2` | PR #1 merged | можно удалить после подтверждения владельцев |
| `origin/codex/mo-ui-source-fallback-plan` | PR #2 squash-merged | можно удалить; ancestry misleading из-за squash |
| `origin/codex/mo-secure-csv-runtime-fix` | PR #3 merged | можно удалить после закрытия worktree |
| `origin/codex/main-sync` | ancestor `main`, устарела | больше не использовать, затем удалить |
| `origin/codex/kz-evaluation-quality-v3` | ancestor `main` | кандидат на удаление |
| `origin/codex/product-ux-redesign-v1` | ancestor `main` | кандидат на удаление |
| `origin/docs/session-plan-2026-07-20` | ancestor `main` | кандидат на удаление |
| `origin/feature/mo-bi-imac` | ancestor `main` | кандидат на удаление |
| local `ci-workflow` | patch-equivalent commit уже в `main` | local branch можно удалить после проверки |

Удаление не выполнено: это destructive housekeeping, требующее подтверждения, что другой
компьютер или агент больше не использует соответствующий worktree.

## 4. Критическое состояние корневого checkout этого компьютера

`/Users/pavelkuzauka/Cursor_Folders/Protocol` нельзя использовать для новой работы:

- local `main` behind `origin/main` на 67 commits;
- local `main` ahead на commit `00e0a4b9` (`.gitignore`, 47 строк);
- 23 modified/deleted tracked files;
- 13 untracked paths.

Не выполнять там `pull`, `reset`, `clean`, `checkout --` или массовый add. Сначала отдельной
задачей:

1. создать rescue-ветку для `00e0a4b9` и проверить, нужен ли этот `.gitignore` diff;
2. инвентаризировать 36 dirty/untracked paths по владельцам;
3. сохранить нужное отдельными тематическими ветками;
4. только после явного подтверждения восстановить clean local `main` из `origin/main`.

До этого все задачи запускать через clean worktree от `origin/main`.

## 5. Новые координационные документы

- `AGENTS.md` - обязательные правила для всех совместимых агентов;
- `.cursor/rules/repository-coordination.mdc` - alwaysApply правило Cursor;
- `docs/deploy/multi-agent-single-repo-render-runbook-v2.md` - подробный workflow;
- `docs/deploy/two-computers-daily-checklist.md` - короткий preflight/checklist;
- старый `multi-machine-git-deploy-runbook.md` помечен superseded.

Ключевое изменение: больше нет общей рабочей `codex/main-sync` и штатного прямого promote
feature HEAD в `main`. Используются отдельные task-ветки, PR и один release-координатор.

## 6. Что ещё нужно сделать

Приоритет P0:

1. Merge/deploy ветки с `r13`; затем добавить Git SHA в `/api/version`.
2. Исправить `report_reconciliation_mismatch` для отсутствующих/расходящихся дневных отчётов.
3. Разобрать глобальный CI: `ruff check .` падает на 102 legacy errors вне МО diff.
4. Настроить branch protection `main`: PR required, no force-push, review/check policy.
5. Настроить auto-delete merged branches.

Приоритет P1:

1. Вынести Render deploy в один GitHub Action с `concurrency: production-render`.
2. Добавить `RENDER_API_KEY` как GitHub secret и не разносить его по компьютерам.
3. Добавить automated post-deploy smoke для health/capabilities/source coverage.
4. Health должен показывать не только наличие secure CSV, но и coverage сопоставления
   warehouse -> clinical source.

## 7. Следующий безопасный старт на другом компьютере

```bash
cd /path/to/Protocol
git status --short --branch
git fetch --prune origin
git rev-list --left-right --count origin/main...HEAD
gh pr list --repo akuazuk/protocol --state open
scripts/ops/git_task_start.sh mo-version-git-sha --pc=pc2 \
  --branch=codex/mo-version-git-sha-agent2-pc2
```

Перед изменением файлов прочитать корневой `AGENTS.md` полностью.
