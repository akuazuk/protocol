# Multi-agent release guard: только merged origin/main в Render (v2)

Дата: 2026-08-03
Статус: active
Предшественник: `2026-07-28-multi-machine-git-deploy-workflow-v1.md`

## Контекст

Production снова собирает `be172d73`, потому что это текущий HEAD `origin/main`.
Новый UI находится в draft PR и ещё не является production-кодом. При этом в репозитории
сохранились старые скрипты, способные продвинуть task HEAD прямо в `main`, а низкоуровневый
Render wrapper по умолчанию принимал локальный HEAD. Документация запрещает этот путь,
но технической блокировки не было.

## Что изменено в production

До merge этой задачи production остаётся на `be172d73` /
`2026-08-03-r19-rubric-handoff-workflow`. Изменения этой задачи управляют release-процессом
и добавляют точный Git SHA в `/api/version`; они ещё не развёрнуты.

## Метрики

- Прямой push task HEAD в `main`: было возможно скриптом, стало 0 штатных путей, цель 0.
- Deploy произвольного/локального SHA: было возможно, стало 0 путей в wrappers, цель 0.
- Canonical release-команда: было несколько, стала 1, цель 1.
- Branch protection `main`: было выключено, стало включено с обязательным PR, цель включено.
- Видимость Git SHA в `/api/version`: было 0, стало 1 поле, цель 1.

## Шаги

- [x] Подтвердить `origin/main = be172d73` и production `r19`.
- [x] Заблокировать старые promote task-HEAD scripts.
- [x] Добавить release wrapper, принимающий только точный HEAD `origin/main`.
- [x] Проверять expected version из release commit, а не из локальной task-ветки.
- [x] Добавить always-applied правило для всех Cursor-агентов и обновить `AGENTS.md`.
- [x] Добавить Git SHA в `/api/version`.
- [x] Включить branch protection для `main`.
- [x] Поднять `BUILD_VERSION` до `2026-08-03-r21-render-main-guard`.
- [x] Завершить тесты, commit, push и открыть PR #7.

## Риски

- Старые клоны до обновления сохраняют опасные scripts. GitHub branch protection должна
  блокировать их прямой push в `main`.
- Обязательный красный baseline CI заблокирует все PR, поэтому до исправления baseline
  защита требует сам PR, но не делает падающий lint обязательным check.
- Render Dashboard остаётся административным обходным путём. Ручной deploy допускается
  только release-координатором после сверки SHA с `origin/main`.
