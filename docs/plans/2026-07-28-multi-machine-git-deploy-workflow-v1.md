# План: workflow для 2 компьютеров (safe pull + deploy guard)

Дата: 2026-07-28
Статус: active

## Контекст

Проект ведется параллельно с двух компьютеров, коммиты и push делаются в разное время.
Риск: diverged `main`, грязные локальные деревья, случайный deploy не из той ветки.

## Цели

1. Перед pull всегда выполнять одинаковый preflight.
2. Стандартизировать запуск сессии на любом ПК.
3. Блокировать deploy при грязном дереве, рассинхроне ветки или неверной ветке.
4. Зафиксировать единый runbook для handoff между ПК.

## Метрики (было / стало / цель)

| Метрика | Было | Стало | Цель |
|---|---:|---:|---:|
| Pull в diverged состоянии без явной диагностики | часто | 0 (блокируется `git_safe_pull.sh`) | 0 |
| Deploy с локальными unpushed коммитами | риск | 0 (блокируется `git_deploy_guard.sh`) | 0 |
| Старт сессии без sync-check | часто | стандартизирован (`git_safe_start.sh`) | 100% |

## Шаги

- [x] Добавить `scripts/git_safe_start.sh`.
- [x] Добавить `scripts/git_safe_pull.sh`.
- [x] Добавить `scripts/git_deploy_guard.sh`.
- [x] Добавить `scripts/git_task_start.sh` (авто-ветка + clean worktree для новой задачи).
- [x] Добавить `scripts/deploy_after_push.sh` и `scripts/render_wait_version.sh` (push + guard + ожидание версии Render).
- [x] Добавить runbook `docs/deploy/multi-machine-git-deploy-runbook.md`.
- [x] Проверить синтаксис скриптов (`bash -n`) и исполняемость.
- [ ] После user-аппрува: прогнать сценарии на обоих ПК и при необходимости уточнить allowlist веток.

## Риски

1. Локальные пользовательские привычки могут обходить wrapper-скрипты.
2. Allowlist веток для deploy может требовать расширения под release-процесс.
3. Скрипт не выполняет merge/rebase автоматически в diverged сценарии - это сознательная
   защита от ошибочных автоматических действий.

## Команды продолжения

```bash
scripts/git_safe_start.sh
scripts/git_safe_pull.sh
scripts/git_deploy_guard.sh --prod-url=https://protocol-bimy.onrender.com
```
