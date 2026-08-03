# Protocol: обязательные правила для агентов и компьютеров

Этот файл - канонический preflight для любой работы в репозитории Protocol. Его нужно
прочитать целиком до изменения файлов, запуска Git-команд с записью или обращения к
production. Cursor дополнительно загружает `.cursor/rules/repository-coordination.mdc`.

Подробный workflow: `docs/deploy/multi-agent-single-repo-render-runbook-v2.md`.

## 1. Источники истины

- Код и история: `origin/main` на `https://github.com/akuazuk/protocol.git`.
- Production: Render service `protocol`, id `srv-d78he6h5pdvs73b1kufg`, домен
  `https://protocol-bimy.onrender.com`.
- `render.yaml` описывает другой, приостановленный сервис `protocol-rag`; он не является
  источником настроек production.
- Данные МО production: `/var/data/medical_exams`. Клинические тексты, токены и ID не
  печатать в логах, PR, handoff и ответах.

## 2. Обязательный preflight каждой сессии

```bash
git status --short --branch
git fetch --prune origin
git rev-list --left-right --count origin/main...HEAD
gh pr list --repo akuazuk/protocol --state open
```

Затем прочитать:

1. этот `AGENTS.md`;
2. актуальный план из `docs/plans/README.md`;
3. последний релевантный handoff из `docs/reports/`;
4. `docs/deploy/two-computers-daily-checklist.md` перед Git/Render-операциями.

Если checkout грязный, отстаёт или расходится с `origin/main`, его не чинят pull/rebase/reset
во время задачи. Создают новый clean worktree от `origin/main`.

## 3. Одна задача - одна ветка - один владелец - один worktree

Никогда не делить одну рабочую ветку между агентами или компьютерами. Имя ветки должно
показывать задачу и владельца, например:

```text
codex/mo-report-reconcile-agent1-pc1
cursor/mo-chart-a11y-agent2-pc2
hotfix/mo-source-runtime-release-pc1
```

Создание worktree:

```bash
scripts/ops/git_task_start.sh <task-slug> --pc=<pc-id> \
  --branch=codex/<task-slug>-<agent-id>-<pc-id>
```

Правила:

- base только свежий `origin/main`;
- не работать напрямую в `main` и не переиспользовать `codex/main-sync`;
- не переключать ветку чужого worktree;
- перед правкой проверить, нет ли открытого PR по тем же файлам;
- коммиты небольшие и тематические; push - только текущей task-ветки;
- после push открыть PR; `main` меняется только merge через GitHub.

## 4. Синхронизация параллельных задач

GitHub - единственный общий координационный слой. Draft PR можно открыть сразу после
первого безопасного commit и указать в нём:

- владельца и компьютер;
- изменяемые каталоги;
- зависимости от других PR;
- тесты и состояние deploy;
- запрет merge до снятия draft.

Перед merge снова выполнить `git fetch origin` и синхронизировать task-ветку с текущей
`origin/main`. Force-push, `reset --hard`, `clean -fd` и постоянный обмен через stash
запрещены. Если два PR меняют один файл, второй владелец ждёт merge первого и переносит
свои изменения на новый `origin/main`.

## 5. Commit, версия и PR

- Перед каждым release-значимым изменением выполнить
  `scripts/ops/bump_build_version.sh <slug>`.
- `/api/version` показывает `BUILD_VERSION`, а не Git SHA. Если версию не поднять, Render
  может развернуть новый код, продолжая показывать старое `rN`.
- До commit: релевантные тесты, `git diff --check`, syntax/lint изменённых файлов.
- Глобальный красный CI нельзя молча игнорировать: в PR отделить baseline-ошибки от ошибок
  diff. Merge при красном обязательном CI требует явного решения владельца.
- После merge task-ветка больше не используется для новой работы.

## 6. Единственный безопасный production workflow

Обычный путь:

1. task-ветка -> PR;
2. проверки и review;
3. squash/merge PR в `origin/main`;
4. release-координатор фиксирует точный merge SHA;
5. только release-координатор запускает/контролирует Render deploy этого SHA;
6. проверяются `/api/version`, `/health` и feature-specific smoke-test.

Нельзя параллельно запускать два deploy, менять Render env или загружать `/var/data`.
Команды `render_promote_main.sh` и `deploy_promote_main_after_push.sh`, продвигающие HEAD
напрямую в `main`, не являются штатным multi-agent workflow; использовать их можно только
как явно согласованную аварийную процедуру после проверки точного SHA.

Если настроен `RENDER_API_KEY`:

```bash
scripts/ops/render_deploy.sh ensure-deploy --commit=<merge-sha> --wait
```

Без API key нужно дождаться webhook deploy и подтвердить новую сборку не только строкой
версии, но и поведением нового endpoint/функции.

## 7. Handoff в конце каждой сессии

Создать или обновить `docs/reports/YYYY-MM-DD-handoff-<topic>.md`:

- repo, branch, worktree, base SHA, HEAD SHA, PR;
- сделано и не сделано;
- тесты и известные baseline failures;
- был ли merge/deploy и точный production SHA;
- `BUILD_VERSION` и результаты smoke-test;
- одна безопасная следующая команда;
- перечень файлов, которые нельзя трогать параллельно.

Не писать «готово», если commit существует только локально, PR не merged или production
не проверен. Эти состояния отмечаются отдельно.

