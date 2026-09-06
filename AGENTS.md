# Protocol: обязательные правила для агентов и компьютеров

Этот файл - канонический preflight для любой работы в репозитории Protocol. Его нужно
прочитать целиком до изменения файлов, запуска Git-команд с записью или обращения к
production. Cursor дополнительно загружает `.cursor/rules/repository-coordination.mdc`.

Подробный workflow: `docs/deploy/multi-agent-workflow-v3.md` (канон параллельной
работы). Runbook v2 под Render - superseded, его разделы про деплой не выполнять.

## 1. Источники истины

- Код и история: `origin/main` на `https://github.com/akuazuk/protocol.git`.
- **Production - единственный контур, GCE.** Project `protocol-home-e1`, VM `protocol-app`
  (`europe-central2-a`), UI `https://protocol.kravira.by` (Caddy → `127.0.0.1:8000`).
  Данные МО: `/var/data/medical_exams` на GCE PD. Инвентарь: `deploy/gcp-app/INVENTORY.md`.
  **Runbook релиза, проверки и откатa: `docs/deploy/gce-production-runbook.md`.**
  Деплой: `bash deploy/gcp-app/deploy_to_gce.sh` релиз-координатором (HEAD ровно на
  `origin/main`; скрипт иначе отказывает). Образ тегируется по SHA, откат - на предыдущий тег.
- **Render продом не является и откатом служить не может:** сервис `protocol`
  (`srv-d78he6h5pdvs73b1kufg`, `protocol-bimy.onrender.com`) приостановлен и отдаёт `503`
  (проверено 2026-09-05). Скрипты `scripts/ops/render_*` оставлены как legacy и блокируют
  деплой без явного `ALLOW_LEGACY_RENDER=1`. Откат делается образом по SHA на GCE и
  снапшотом диска, а не переключением на Render.
- Порт `8000` наружу закрыт: firewall разрешает только `80/443`, контейнер публикуется на
  `127.0.0.1:8000`. Не открывать снова - до 2026-09-05 всё приложение отдавалось по
  plaintext HTTP в обход TLS и HSTS.
- `render.yaml` описывает ещё один приостановленный сервис `protocol-rag`; источником
  настроек он тоже не является.
- Клинические тексты, токены и ID не печатать в логах, PR, handoff и ответах.
- **Gemini / night LLM для МО** - не с Mac. Primary: `deploy/gcp-llm/run_on_gce.sh`.
  Legacy Render: `scripts/run_mo_render_llm_backfill.sh` (VanyaVPN `ensure-off` перед SSH).
  План контуров: `docs/plans/2026-08-07-by-home-gcp-llm-split-v1.md`. Не смешивать MIS DSN в llm-образ.
  MIS extract (E2): только GCE cron 02:00 UTC (+ retry 03:00); пароль MIS в
  Secret Manager `kravira-db-password`; non-secret DSN в `/opt/protocol/.env.mis`
  (owner cron user `pavel`). Mac launchd SQL выключен.

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

Параллельные вкладки не делят ветку и не делят worktree. Перед правкой и после чужого
merge:

```bash
scripts/ops/check_pr_file_overlap.sh
scripts/ops/rebase_task_onto_main.sh
```

CI одной PR не отменяет прогон другой и не убивает уже идущий run той же ветки
(`cancel-in-progress: false`). Конфликт только в `BUILD_VERSION` снимает rebase-скрипт
сам; любой другой конфликт - стоп и разделение файлов. Workflow `PR overlap notify`
пишет комментарий, но не является required check.

## 5. Commit, версия и PR

- Перед каждым release-значимым изменением выполнить
  `scripts/ops/bump_build_version.sh <slug>`.
- `/api/version` показывает `BUILD_VERSION` **и** `git_commit`. Если версию не поднять,
  прод развернёт новый код, продолжая показывать старую версию, и деплой-скрипт
  остановит релиз с откатом (он сверяет `version` с `BUILD_VERSION` релизного коммита).
- До commit: релевантные тесты, `git diff --check`, syntax/lint изменённых файлов.
- **Без AI/vendor attribution:** не писать в коммитах, PR, docs и коде трейлеры
  `Co-authored-by: …` от IDE/агентов, `Made with …`, `Generated by …` и аналогичные
  брендовые пометки. Сообщение коммита - только суть изменения.
- Глобальный красный CI нельзя молча игнорировать: в PR отделить baseline-ошибки от ошибок
  diff. Merge при красном обязательном CI требует явного решения владельца.
- После merge task-ветка больше не используется для новой работы.

## 6. Единственный безопасный production workflow

Обычный путь (код):

1. task-ветка -> PR;
2. проверки и review;
3. squash/merge PR в `origin/main`;
4. release-координатор фиксирует точный merge SHA.

**Единственный прод-путь (GCE):** после merge, если затронуты runtime/UI/MO,
релиз-координатор запускает:

```bash
bash deploy/gcp-app/deploy_to_gce.sh
# smoke:
curl -fsS https://protocol.kravira.by/health/live
curl -fsS https://protocol.kravira.by/api/version   # version + ожидаемый BUILD_VERSION
```

Скрипт сам отказывается работать, если `HEAD != origin/main`; уезжает `git archive`
релизного SHA, а не рабочее дерево; образ тегируется по SHA; при неудачном health или
несовпадении `version`/`git_commit` выполняется автоматический откат на предыдущий образ.
Подробности и ручной откат: `docs/deploy/gce-production-runbook.md`.

Автодеплоя из GitHub после merge **пока нет**: workflow
`.github/workflows/gce-production-deploy.yml` подготовлен, но требует Workload Identity
Federation и deploy-сервисного аккаунта (переменные перечислены в заголовке workflow).
Включать - отдельным решением владельца.

**Render как «backup» больше не рассматривается:** сервис приостановлен (`503`), Action
`Production Render release` удалён. Предлагать «дождаться Render deploy» после merge -
ошибка. Откат делается образом по SHA на GCE плюс снапшот `protocol-data`.

Нельзя параллельно менять GCE `/var/data`, firewall или env без координатора. Прямой push
в `main` запрещён. `render_promote_main.sh` и promote task-HEAD отключены.
Deploy считается завершённым, когда на `https://protocol.kravira.by` совпали
`version` / `git_commit`, `/health/live` ok и пройден feature smoke.

## 7. Handoff в конце каждой сессии

Создать или обновить `docs/reports/YYYY-MM-DD-handoff-<topic>.md`:

- repo, branch, worktree, base SHA, HEAD SHA, PR;
- сделано и не сделано;
- тесты и известные baseline failures;
- был ли merge/deploy и точный production SHA;
- `BUILD_VERSION` и результаты smoke-test;
- одна безопасная следующая команда;
- перечень файлов, которые нельзя трогать параллельно.

Не писать «готово», если commit существует только локально, PR не merged или primary GCP
не проверен (`protocol.kravira.by`). Эти состояния отмечаются отдельно.

**Сейчас (2026-08-08):** primary UI/данные - GCP `https://protocol.kravira.by`; Render -
backup. План: `docs/plans/2026-08-07-by-home-gcp-llm-split-v1.md`. Перед ночным cutover
Mac launchd → extract fallback-only; MIS + score/LLM на GCE.

