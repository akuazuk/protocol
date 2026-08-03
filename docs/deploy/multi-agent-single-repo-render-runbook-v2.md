# Runbook v2: несколько агентов, компьютеров и один Render production

Дата: 2026-08-03
Статус: active, canonical

Короткие обязательные правила находятся в корневом `AGENTS.md`. Этот документ объясняет
операционный workflow и восстановление из текущего состояния репозитория.

## 1. Почему прежняя модель стала опасной

В старой модели несколько сессий использовали `codex/main-sync`, а затем напрямую
продвигали HEAD рабочей ветки в `main`. При параллельной работе это создаёт риски:

- один агент может затереть смысл коммитов другого;
- squash-merged ветка выглядит «не merged» по ancestry, хотя PR уже закрыт;
- локальный `main` может иметь собственный commit и десятки грязных файлов;
- два агента могут одновременно запустить разные Render deploy;
- неизменённый `BUILD_VERSION` скрывает факт новой раскатки.

Поэтому общий mutable branch заменяется task-ветками и PR, а deploy сериализуется.

## 2. Роли

- **Владелец задачи** меняет только свою task-ветку и открывает PR.
- **Ревьюер** проверяет diff, тесты, PHI/secrets и влияние на соседние PR.
- **Release-координатор** один на конкретный merge/deploy; фиксирует merge SHA и проверяет
  production. Остальные агенты в это время не деплоят и не меняют Render env/data.

Один человек может выполнять все роли последовательно, но не несколько deploy одновременно.

## 3. Начало работы на любом компьютере

Нельзя начинать с `git pull` в неизвестном checkout.

```bash
cd /path/to/Protocol
git status --short --branch
git fetch --prune origin
git rev-list --left-right --count origin/main...HEAD
gh pr list --repo akuazuk/protocol --state open
```

Создать новый worktree:

```bash
scripts/ops/git_task_start.sh mo-example --pc=pc2 \
  --branch=codex/mo-example-agent2-pc2
cd /private/tmp/protocol-task-mo-example-pc2
```

Проверка:

```bash
test -z "$(git status --porcelain)"
git merge-base --is-ancestor origin/main HEAD
```

## 4. Владение файлами и конфликтами

Перед правкой посмотреть открытые PR и их changed files. Если другой PR меняет тот же
backend/UI файл, выбрать одно:

1. разделить задачу по непересекающимся файлам;
2. дождаться первого merge и создать ветку заново от обновлённого `origin/main`;
3. явно назначить одного владельца интеграционного PR.

Не пересылать незакоммиченные изменения между компьютерами. Общий checkpoint - commit в
task-ветке на origin. Токены, `.env`, дампы БД, clinical CSV/parquet и PHI не коммитить.

## 5. Проверки и PR

```bash
scripts/ops/bump_build_version.sh <release-slug>  # если изменение попадёт в deploy
git diff --check
# релевантные tests/lint/syntax
git add <точные-файлы>
git commit -m "type(scope): summary"
git push -u origin HEAD
```

PR должен содержать base/head SHA, scope, тесты, baseline failures, migration/data impact и
production smoke plan. Draft PR используется как объявление владения задачей.

Перед merge:

```bash
git fetch origin
git rev-list --left-right --count origin/main...HEAD
```

При изменившемся `main` task-ветку синхронизирует её владелец. Не разрешать конфликт через
выбрасывание чужих блоков и не использовать force-push shared веток.

## 6. Как понимать merged ветки после squash

После squash GitHub создаёт новый commit в `main`. Исходные commits task-ветки могут не
быть ancestors `main`, поэтому `git branch --merged` и граф истории дают ложное ощущение,
что ветка не вошла.

Каноническая проверка:

1. PR имеет `merged=true`;
2. merge/squash SHA присутствует в `origin/main`;
3. diff PR есть в `main`;
4. production smoke подтверждает функцию.

Только после этого и подтверждения, что агент больше не использует worktree, ветку можно
удалять. Удаление ветки не удаляет merged код.

## 7. Production deploy

Production service:

```text
name: protocol
service id: srv-d78he6h5pdvs73b1kufg
url: https://protocol-bimy.onrender.com
branch: main
```

`protocol-rag` из `render.yaml` - другой приостановленный сервис. Не resume/deploy его по
ошибке.

Release-координатор после merge:

```bash
git fetch origin
merge_sha=$(git rev-parse origin/main)
scripts/ops/git_deploy_guard.sh --render-git --render-branch=main \
  --prod-url=https://protocol-bimy.onrender.com
scripts/ops/render_deploy.sh ensure-deploy --commit="$merge_sha" --wait
```

Если `RENDER_API_KEY` отсутствует, не имитировать успешный deploy. Дождаться webhook и
проверить production endpoint. В любом случае:

```bash
curl -fsS https://protocol-bimy.onrender.com/api/version
curl -fsS https://protocol-bimy.onrender.com/health/live
```

Для защищённых МО endpoints токен брать только из локального `.env`, передавать как
`X-Methodist-Token`, не печатать его и не выводить case ID/clinical text.

## 8. Один deploy за раз

Перед deploy release-координатор объявляет точный SHA в PR/handoff. Другие агенты ждут
завершения. Во время окна запрещено:

- merge второго release PR;
- manual deploy/restart другого SHA;
- изменение Render environment;
- загрузка/перестройка `/var/data/medical_exams`.

Deploy завершён только после статуса live и smoke-test. Строка `BUILD_VERSION` обязательна,
но недостаточна: проверяется новая функция, потому что версия может быть забыта.

## 9. Восстановление грязного или diverged checkout

Не выполнять reset/clean/checkout. Сначала зафиксировать:

```bash
git status --short --branch
git rev-list --left-right --count origin/main...HEAD
git log --oneline origin/main..HEAD
```

Локальные commits сохранить отдельной rescue-веткой и push только после просмотра diff.
Незакоммиченные файлы инвентаризировать отдельно. До решения владельца продолжать работу в
новом clean worktree от `origin/main`.

## 10. Рекомендуемые настройки GitHub/Render

Следующие изменения требуют отдельного административного решения:

1. branch protection для `main`: PR required, no force-push, required review;
2. исправить CI baseline, чтобы required check снова был значимым;
3. включить auto-delete head branches после merge;
4. добавить Git SHA/deploy time в `/api/version` наряду с `BUILD_VERSION`;
5. перенести deploy в один GitHub Action с `concurrency: production-render`;
6. хранить Render API key как secret CI, а не копировать между компьютерами.

