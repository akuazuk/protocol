# Ежедневный чеклист: несколько агентов и компьютеров

Канонические правила: `AGENTS.md` и
`docs/deploy/multi-agent-single-repo-render-runbook-v2.md`.

## 1. Preflight - до любой правки

```bash
git status --short --branch
git fetch --prune origin
git rev-list --left-right --count origin/main...HEAD
gh pr list --repo akuazuk/protocol --state open
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

## 4. Merge и deploy - только release-координатор

Штатный путь: task branch -> PR -> squash/merge -> exact SHA в `origin/main` -> Render.

```bash
git fetch origin
merge_sha=$(git rev-parse origin/main)
scripts/ops/render_deploy.sh ensure-deploy --commit="$merge_sha" --wait
```

`render_promote_main.sh` и `deploy_promote_main_after_push.sh` не использовать в обычной
параллельной работе: они предназначены только для явно согласованной аварийной процедуры.

Production:

- service: `protocol` / `srv-d78he6h5pdvs73b1kufg`;
- URL: `https://protocol-bimy.onrender.com`;
- `protocol-rag` из `render.yaml` - не production.

## 5. Проверка production

```bash
curl -fsS https://protocol-bimy.onrender.com/api/version
curl -fsS https://protocol-bimy.onrender.com/health/live
```

Кроме версии обязательно проверить изменённую функцию. Для МО использовать локальный
`METHODIST_TOKEN` только в `X-Methodist-Token`; не выводить токен, case ID и clinical text.

## 6. Handoff

Записать в `docs/reports/YYYY-MM-DD-handoff-<topic>.md`: branch, worktree, base/head SHA,
PR, тесты, merge SHA, deploy, `BUILD_VERSION`, smoke-test, оставшиеся задачи и одну следующую
безопасную команду.
