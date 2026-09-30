# Handoff: инструкция агенту на другом компьютере

Дата: 2026-09-30

## Repo

- repo: `akuazuk/protocol`
- branch: `cursor/second-computer-agent-brief-agent1-pc1`
- worktree: `/private/tmp/protocol-task-second-computer-agent-brief-pc1`
- base до инструкции: `e2d87891` (#338)
- merge в `origin/main`: `9e497c5e` (#339)
- PR: https://github.com/akuazuk/protocol/pull/339
- инструкция: `docs/deploy/second-computer-agent.md`

## Сделано

- Полная инструкция старта на новом компьютере: clone, worktree, запреты, снимок открытых PR, почему Docker на прод-VM не заменяет второй компьютер.
- Ссылки из `AGENTS.md`, `docs/deploy/multi-agent-workflow-v3.md`, `docs/deploy/two-computers-daily-checklist.md`, `.cursor/rules/next-chat-handoff.mdc`.
- `BUILD_VERSION`: `2026-09-30-175542Z-second-pc-agent` (только эта строка в `rag_server.py`).

## Не сделано

- Deploy. Документ не меняет runtime. Инструкция уже в `origin/main` (`9e497c5e`).
- Открытые PR не влиты сознательно. См. раздел 8 инструкции: #261 и #186 `DIRTY`, #113 клинический и `DIRTY`, Dependabot на Python 3.14 и зависимости с красной гигиеной или тестами.
- Каталог `/Users/pavelkuzauka/Cursor_Folders/Protocol` не синхронизирован: локальный `main` отстаёт, в нём старый индекс планов и незакоммиченные файлы, которые на `origin/main` уже есть. `pull` / `reset` / `clean` там не делать.

## Тесты

- `git diff --check` без замечаний.
- Продуктовые тесты не запускались: diff - документация и строка версии.

## Прод

- Deploy не выполнялся. `protocol.kravira.by` эту версию ещё не показывает.
- Smoke прода для чтения инструкции не нужен: новый компьютер берёт файл из git.

## Одна безопасная следующая команда

```bash
git clone https://github.com/akuazuk/protocol.git
# прочитать docs/deploy/second-computer-agent.md
```

## Не трогать параллельно, пока открыты чужие PR

- `docs/plans/README.md` (#261, #186, #113)
- `rag_server.py` кроме строки `BUILD_VERSION` (#186, #113)
- `deploy/gcp-llm/run_on_gce.sh`, `eval/mo_score_calibration/` (#113)
- Dockerfile в `deploy/*` (#194-#197)
