# Runbook: безопасная работа с двух компьютеров

Дата: 2026-07-28
Статус: active

Этот runbook фиксирует единый workflow для двух компьютеров, чтобы:
- не ломать `main` в грязном состоянии;
- перед pull всегда проходить preflight;
- деплоить только из нужных веток;
- исключить рассинхрон и случайные конфликты.

## 1) Единый принцип

1. Не работаем в грязном `main`.
2. Для активной разработки используем отдельные ветки и clean worktree.
3. Перед pull всегда запускаем проверку `scripts/git_safe_pull.sh`.
4. Перед deploy всегда запускаем `scripts/git_deploy_guard.sh`.

## 2) Старт сессии на любом компьютере

```bash
cd /path/to/Protocol
scripts/git_safe_start.sh
```

Если `main` грязный или расходится:

```bash
AUTO_WORKTREE=1 scripts/git_safe_start.sh --auto-worktree
cd /private/tmp/protocol-main-sync
```

### 2.1. Автостарт новой задачи (рекомендуется)

```bash
scripts/git_task_start.sh mo-daily-fix --pc=pc1
```

Что делает:
- `fetch origin`;
- создаёт отдельный clean worktree от `origin/main`;
- создаёт ветку вида `feature/<task>-pcX`;
- выводит следующие команды, чтобы не запутаться.

## 3) Безопасный pull

В текущей рабочей ветке:

```bash
scripts/git_safe_pull.sh
```

Скрипт:
- останавливает pull при грязном дереве;
- делает `fetch`;
- делает `pull --ff-only`, если можно;
- при diverged состоянии не выполняет опасных действий и печатает точные шаги.

## 4) Рекомендованная модель веток для 2 ПК

- `feature/<topic>-pc1`
- `feature/<topic>-pc2`

Правила:
- одна задача - одна ветка;
- не переиспользовать ветку для другой задачи;
- после каждой логической порции работы сразу `git push`.

## 5) Pre-deploy guard

Перед deploy на любом ПК:

```bash
scripts/git_deploy_guard.sh --prod-url=https://protocol-bimy.onrender.com
```

Проверяется:
- ветка в allowlist (`main release/* hotfix/* codex/main-sync`);
- чистое рабочее дерево;
- нет unpushed коммитов;
- нет отставания от upstream;
- валидный формат `BUILD_VERSION`.

Если guard упал, deploy не выполняем, сначала исправляем причину.

### 5.1. Если Render деплоит напрямую из Git

Если в Render сервис подключён к git-ветке (обычно `main`), используйте строгий режим:

```bash
scripts/git_deploy_guard.sh --render-git --render-branch=main \
  --prod-url=https://protocol-bimy.onrender.com
```

Этот режим дополнительно блокирует deploy, если текущая ветка не совпадает с веткой,
связанной с Render.

Для максимальной простоты можно запускать wrapper:

```bash
scripts/deploy_after_push.sh --branch=main --prod-url=https://protocol-bimy.onrender.com
```

Он делает `git push` и сразу запускает строгий guard для Render Git-ветки.

## 6) Минимальный handoff между ПК

После каждой сессии фиксируем:
- ветка;
- commit SHA;
- что сделано;
- что осталось;
- какую команду запускать следующей.

Рекомендуемый путь:

`docs/reports/YYYY-MM-DD-handoff-<topic>.md`

## 7) Анти-паттерны (запрещено)

- `git reset --hard`
- `git clean -fd`
- `git checkout -- <file>`
- `git stash` как постоянная стратегия синхронизации между ПК
- force-push в shared ветки

## 8) Быстрый чеклист (копипаст)

```bash
# start
scripts/git_safe_start.sh

# if needed
AUTO_WORKTREE=1 scripts/git_safe_start.sh --auto-worktree
cd /private/tmp/protocol-main-sync

# sync branch
scripts/git_safe_pull.sh

# start new task
scripts/git_task_start.sh <task-slug> --pc=pc1

# work, test, commit, push

# before deploy
scripts/git_deploy_guard.sh --prod-url=https://protocol-bimy.onrender.com
scripts/git_deploy_guard.sh --render-git --render-branch=main --prod-url=https://protocol-bimy.onrender.com
scripts/deploy_after_push.sh --branch=main --prod-url=https://protocol-bimy.onrender.com
```
