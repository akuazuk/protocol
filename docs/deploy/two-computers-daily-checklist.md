# Ежедневный чеклист: несколько агентов и компьютеров

Канонические правила: `AGENTS.md` и
`docs/deploy/multi-agent-workflow-v3.md`.

## 1. Preflight - до любой правки

```bash
git status --short --branch
git fetch --prune origin
git rev-list --left-right --count origin/main...HEAD
python3 scripts/ops/check_branch_alive.py --online
python3 scripts/ops/pr_dashboard.py
```

Guard должен подтвердить, что task-ветка жива и не была удалена после merge.
Не обходить его через `ALLOW_DEAD_BRANCH=1` или `--no-verify` без
диагностированной ошибки guard. Последняя команда заменяет ручной `gh pr list`:
она показывает не только список PR, но и кто какие файлы держит, что с чем
жёстко пересекается и какие PR зависли. Перед правкой конкретного файла:

```bash
python3 scripts/ops/pr_dashboard.py --files <path>
```

Если checkout грязный, behind или diverged - не выполнять pull/reset/clean. Создать clean
task-worktree:

```bash
scripts/ops/git_task_start.sh <task-slug> --pc=<pc-id> \
  --branch=codex/<task-slug>-<agent-id>-<pc-id>
```

## 2. Работа

- одна задача, ветка, владелец и worktree;
- base - свежий `origin/main`;
- проверить открытые PR на пересечение файлов;
- не использовать общую `codex/main-sync`;
- не работать и не коммитить напрямую в `main`.

## 3. Проверка и публикация

```bash
scripts/ops/bump_build_version.sh <slug>  # для release-значимого изменения
git diff --check
# запустить релевантные тесты/lint/syntax
git add <точные-файлы>
git commit -m "type(scope): summary"
git push -u origin HEAD
```

Открыть PR. Перед merge обновить refs и проверить расхождение с `origin/main`. Красный
required CI нельзя игнорировать без явного решения владельца.

## 4. Merge и deploy — только release-координатор

Task PR → актуальный main → обязательные проверки → squash/merge → exact main SHA.
Опубликованную историю не переписывать: merge main либо новый task PR, см. workflow v3.
Один координатор и одно окно. Не merge другой runtime PR во время release.

Единственный production — GCE, `https://protocol.kravira.by`:

```bash
bash deploy/gcp-app/deploy_to_gce.sh
```

HEAD должен ровно совпадать со свежим origin/main; новый release worktree и
подробные условия описаны в [GCE runbook](gce-production-runbook.md).
Render приостановлен и не является продом или откатом. Автодеплой GitHub пока
не настроен; не ждать его после merge. Откат — предыдущий образ по SHA на GCE.

## 5. Проверка production

```bash
curl -fsS https://protocol.kravira.by/health/live
curl -fsS https://protocol.kravira.by/api/version
```

Сверить BUILD_VERSION и git_commit с release SHA, затем проверить изменённую
функцию. Для МО токен передавать только в X-Methodist-Token, не выводить токен,
идентификаторы пациентов/визитов и клинический текст. Локальный synthetic test
не заменяет feature smoke, но production smoke не должен изменять данные без задачи.

## 6. Handoff и сохранность worktree

Записать в docs/reports/YYYY-MM-DD-handoff-topic.md: branch/worktree, base/head,
PR, тесты, merge SHA, production SHA/version/smoke, оставшиеся задачи,
файлы под владением и одну безопасную следующую команду.

Активные worktree хранить вне временных каталогов и защищать git worktree lock.
Cleanup не снимает lock и не удаляет чужую работу. Удалять свой завершённый
worktree только после проверки публикации, merge и handoff; чужой — после
согласования с владельцем. Исторические handoff описывают прошлый контур;
текущие AGENTS.md и GCE runbook имеют приоритет.
