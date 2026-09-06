# Handoff: единые инструкции GCE и синхронизации

2026-09-06; akuazuk/protocol; agent1 / pc1.
Branch codex/gce-coordination-docs-agent1-pc1.
Worktree /Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/gce-coordination-docs (locked).
Base fe0734a8a5956d1e7a8d494da895319411968d01; HEAD — commit с этим handoff, SHA в PR.

Согласованы AGENTS.md, daily checklist, workflow v3 и GCE runbook:
устранены активные инструкции Render/backup и пример destructive reset.
Release — новый detached worktree точного main, один координатор, без
сопутствующего включения scoring flags. Опубликованная task-ветка обновляется
merge main или новым PR; неопубликованные коммиты допускают rebase.
Active worktree защищён lock и не удаляется чужим cleanup.
Cursor rule ссылается на эти документы; менять его не потребовалось.

Проверены локальные ссылки и bash -n исполняемых блоков без placeholders,
git diff --check. Runtime, CI barriers и deploy scripts не менялись.
BUILD_VERSION не требуется. Merge/deploy нет; последний проверенный production
a592d588. #205 уже merged fe0734a8, не deployed нами.

Держим перечисленные четыре документа и этот handoff. PR #210 меняет другой
handoff, пересечений нет. Исторические отчёты не переписывались. Helper
 git_task_start.sh пока печатает legacy подсказку Render — отдельная ops-задача,
не выполнять её вопреки каноническому GCE runbook.

Следующая безопасная команда:

```bash
gh pr list --repo akuazuk/protocol --state open
```
