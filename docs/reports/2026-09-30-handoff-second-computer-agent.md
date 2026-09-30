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
- `BUILD_VERSION` инструкции: `2026-09-30-175542Z-second-pc-agent`. Уборка PR: `2026-09-30-183156Z-agent-pr-cleanup`.

## Не сделано

- Deploy. `protocol.kravira.by` версию `2026-09-30-183156Z-agent-pr-cleanup` ещё не показывает.
- Каталог `/Users/pavelkuzauka/Cursor_Folders/Protocol` не синхронизирован и рабочим местом не является. `pull` / `reset` / `clean` там не делать.
- Dependabot pip и Actions оставлены открытыми: #198, #199, #200, #201, #203, #248. Пакетом не мержить.

## Закрыто 2026-09-30, ветки сохранены

- #194-#197 Python 3.14 в Dockerfile. Повтор мажора для образа `python` выключен в `.github/dependabot.yml`.
- #261 `cursor/mo-workspace-plan-pc1`
- #186 `cursor/rz-quality-article-layout-agent1-pc1`
- #113 `cursor/mo-calibration-confirmatory-proxy-c9a-pc1`

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

## Не трогать параллельно

- Открытые Dependabot PR #198, #199, #200, #201, #203, #248: не мержить пакетом и не решать их версии внутри другой задачи.
- Закрытые ветки #261, #186, #113 не дописывать. Новая работа по той теме - новая ветка от `origin/main`.
