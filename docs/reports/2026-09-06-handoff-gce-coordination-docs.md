# Handoff: единые инструкции GCE и синхронизации

2026-09-06; akuazuk/protocol; agent1 / pc1.
Branch codex/gce-coordination-docs-agent1-pc1.
Worktree /Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/gce-coordination-docs (locked).
Исходный base fe0734a8a5956d1e7a8d494da895319411968d01. Ветка синхронизирована
merge-коммитом с `origin/main` 246e35336f8b73b3f66e31c38b0d58506c1ce099;
финальный HEAD указан в PR.

Согласованы AGENTS.md, daily checklist, workflow v3 и GCE runbook:
устранены активные инструкции Render/backup и пример destructive reset.
Release — новый detached worktree точного main, один координатор, без
сопутствующего включения scoring flags. Опубликованная task-ветка обновляется
merge main или новым PR; неопубликованные коммиты допускают rebase.
Active worktree защищён lock и не удаляется чужим cleanup.
Cursor rule ссылается на эти документы; менять его не потребовалось. После merge
#216 / 8270b8746e9f3b2e48f9d91feedb87cd6bb8113c канон дополнен обязательной
проверкой `check_branch_alive.py --online` и запретом штатного обхода guard.

До публикации проверить локальные ссылки, bash -n исполняемых блоков без
placeholders, `test_plans_index`, `git diff --check` и required CI актуального
HEAD. Runtime, CI barriers и deploy scripts не менялись.

Runtime batch #205-#208/#211-#213/#212/#215 и packaging #217 уже merged.
Production выпущен из точного `main` 246e35336f8b73b3f66e31c38b0d58506c1ce099:
version `2026-09-06-115045Z-mo-score-availability-ui`, health/MO API/search,
security headers и lab image verification успешны. Во время первого corpus-sync
оборвался SSH; корпус восстановлен штатным sync, проверено 478 PDF и 467 summary,
после чего deploy завершён с `SYNC_PROTOCOL_CORPUS=0`.

Держим перечисленные четыре документа и этот handoff. PR #210 меняет другой
handoff, пересечений нет. Исторические отчёты не переписывались.
`git_task_start.sh` исправлен в #216 и подключает guard автоматически.

Следующая безопасная команда:

```bash
gh pr checks 214 --repo akuazuk/protocol
```
