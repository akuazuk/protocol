# Runbook: безопасная работа с двух компьютеров

Дата: 2026-07-28
Статус: superseded

> Этот документ сохранён как история. Для текущей работы обязательны
> корневой `AGENTS.md` и
> `docs/deploy/multi-agent-workflow-v3.md`. Общая
> `codex/main-sync` и прямой promote task-HEAD в `main` больше не являются
> штатным workflow.

Этот runbook фиксирует единый workflow для двух компьютеров, чтобы:
- не ломать `main` в грязном состоянии;
- перед pull всегда проходить preflight;
- деплоить только из нужных веток;
- исключить рассинхрон и случайные конфликты.

Краткая ежедневная версия в 5 шагов:
`docs/deploy/two-computers-daily-checklist.md`

Примечание: старые команды `scripts/*.sh` сохранены для совместимости, но каноничный путь -
`scripts/ops/*.sh`.

## 1) Единый принцип

1. Не работаем в грязном `main`.
2. Для активной разработки используем отдельные ветки и clean worktree.
3. Перед pull всегда запускаем проверку `scripts/ops/git_safe_pull.sh`.
4. Перед deploy всегда запускаем `scripts/ops/git_deploy_guard.sh`.

## 2) Старт сессии на любом компьютере

```bash
cd /path/to/Protocol
scripts/ops/git_safe_start.sh
```

Если `main` грязный или расходится:

```bash
AUTO_WORKTREE=1 scripts/ops/git_safe_start.sh --auto-worktree
cd /private/tmp/protocol-main-sync
```

### 2.1. Автостарт новой задачи (рекомендуется)

```bash
scripts/ops/git_task_start.sh mo-daily-fix --pc=pc1
```

Что делает:
- `fetch origin`;
- создаёт отдельный clean worktree от `origin/main`;
- создаёт ветку вида `feature/<task>-pcX`;
- выводит следующие команды, чтобы не запутаться.

## 3) Безопасный pull

В текущей рабочей ветке:

```bash
scripts/ops/git_safe_pull.sh
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
scripts/ops/git_deploy_guard.sh --prod-url=https://protocol-bimy.onrender.com
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
scripts/ops/git_deploy_guard.sh --render-git --render-branch=main \
  --prod-url=https://protocol-bimy.onrender.com
```

Этот режим дополнительно блокирует deploy, если текущая ветка не совпадает с веткой,
связанной с Render.

Если работа велась не в `main` (например `codex/main-sync`), перед ожиданием деплоя
нужно продвинуть текущий `HEAD` в `origin/main`:

```bash
scripts/ops/render_promote_main.sh --prod-url=https://protocol-bimy.onrender.com
```

Скрипт делает только fast-forward promote (`HEAD -> origin/main`) и проверяет, что remote
ветка действительно получила ваш SHA.

Для максимальной простоты можно запускать wrapper:

```bash
scripts/ops/deploy_after_push.sh --branch=main --prod-url=https://protocol-bimy.onrender.com
```

Он делает `git push` и сразу запускает строгий guard для Render Git-ветки.

Чтобы дождаться фактического обновления версии в проде, используйте:

```bash
scripts/ops/deploy_after_push.sh --branch=main --prod-url=https://protocol-bimy.onrender.com --wait-version
```

Или отдельной командой:

```bash
scripts/ops/render_wait_version.sh --prod-url=https://protocol-bimy.onrender.com
```

Для режима «одна команда и без ручной путаницы» используйте:

```bash
scripts/ops/deploy_promote_main_after_push.sh --prod-url=https://protocol-bimy.onrender.com
```

Команда рассчитана на работу из любой ветки задачи: сначала `push` текущей ветки, затем
безопасный fast-forward promote в `origin/main` и ожидание новой версии в проде.

Важно: параметры в ops-скриптах передавать как `--key=value` (например
`--prod-url=https://protocol-bimy.onrender.com`), а не через пробел.

### 5.2. Управление сервисом Render через API

История (2026-07-30): авто-деплой по push долгое время **не срабатывал**. В настройках
сервиса стояло `autoDeploy: yes` с триггером `commit`, но за всю доступную историю не было
ни одного деплоя с триггером `new_commit` - только `manual`, то есть клики в дашборде.
Второй сервис того же репозитория (`protocol-rag`) пропустил 159 коммитов в `main` с 1 июля.

Причина: приложение **Render не было установлено как GitHub App** на репозиторий. Оно
числилось только в Authorized GitHub Apps, поэтому Render мог клонировать репозиторий (и
ручные деплои брали свежий код), но события о push ему не приходили. Лечится установкой
https://github.com/apps/render на репозиторий; проверять в
https://github.com/settings/installations, а не в Authorized.

Даже с рабочим webhook push сам по себе не гарантирует раскатку нужного SHA, поэтому
`render_promote_main.sh` и
`deploy_after_push.sh` теперь после push явно запускают деплой нужного коммита через API
(шаг вынесен в `scripts/ops/render_apply_deploy.sh`). Без `RENDER_API_KEY` они
откатываются к старому поведению и просто ждут версию, предупреждая об этом.

Для ручных операций есть `scripts/ops/render_deploy.sh`.

Требуется `RENDER_API_KEY` в `.env` (файл в `.gitignore`, ключ не коммитить).
Создать ключ: https://dashboard.render.com/u/settings?add-api-key

```bash
# почему прод не обновился: autoDeploy, ветка, suspended, последние деплои, версия
scripts/ops/render_deploy.sh status

# логи сборки и рантайма (единственный способ увидеть причину упавшего деплоя)
scripts/ops/render_deploy.sh logs --limit=200

# редеплой без нового коммита, с ожиданием статуса live
scripts/ops/render_deploy.sh deploy --wait

# редеплой с чисткой build cache, если сборка падает на зависимостях
scripts/ops/render_deploy.sh deploy --clear-cache --wait

# перезапуск без пересборки: нужен после заливки данных на /var/data по SSH
scripts/ops/render_deploy.sh restart --wait
```

`status` явно предупреждает, если `autoDeploy: no` - в этом случае push в `main` деплой
не запустит, и нужен `deploy` вручную.

На аккаунте есть второй сервис из того же репозитория - `protocol-rag`
(`srv-d8tb43ojs32c73dd0l00`), созданный Blueprint-синком из `render.yaml` 23 июня.
Прод - это не он, а `protocol` (`srv-d78he6h5pdvs73b1kufg`, домен `protocol-bimy`).
С 2026-07-30 `protocol-rag` приостановлен, чтобы не собираться на каждый push и не
тратить план standard; диск сохранён. Вернуть при необходимости:

```bash
scripts/ops/render_deploy.sh resume --service-id=srv-d8tb43ojs32c73dd0l00
```

### 5.3. Переменные окружения прода

`render.yaml` **не управляет продом** - он описывает приостановленный `protocol-rag`.
Из 59 объявленных там переменных на боевом сервисе стоят 10, а `buildCommand` у него
без установки tesseract (OCR при этом работает: `ocr_image_bytes` откатывается на
Gemini Vision). Переменная, добавленная в `render.yaml`, до прода не доедет.

Настройки прода смотреть и менять так:

```bash
scripts/ops/render_env.sh diff             # что объявлено в render.yaml, но отсутствует на проде
scripts/ops/render_env.sh list             # что реально стоит (значения замаскированы)
scripts/ops/render_env.sh set KEY=VALUE    # применить к проду, Render сам передеплоит
```

Известное расхождение: на проде `RENDER_PLAN=standard`, хотя сервис на плане `pro`.
Из-за этого `_render_high_ram()` в `rag_server.py` возвращает False. Ключевые тумблеры
(`RAG_VECTOR_INDEX`, `RAG_MEMORY_SAVER`) выставлены явно, так что менять это надо
осознанно и отдельной задачей.

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
scripts/ops/git_safe_start.sh

# if needed
AUTO_WORKTREE=1 scripts/ops/git_safe_start.sh --auto-worktree
cd /private/tmp/protocol-main-sync

# sync branch
scripts/ops/git_safe_pull.sh

# start new task
scripts/ops/git_task_start.sh <task-slug> --pc=pc1

# work, test, commit, push

# before deploy
scripts/ops/git_deploy_guard.sh --prod-url=https://protocol-bimy.onrender.com
scripts/ops/git_deploy_guard.sh --render-git --render-branch=main --prod-url=https://protocol-bimy.onrender.com
scripts/ops/deploy_after_push.sh --branch=main --prod-url=https://protocol-bimy.onrender.com
scripts/ops/deploy_after_push.sh --branch=main --prod-url=https://protocol-bimy.onrender.com --wait-version
# if current branch is not main but Render deploys from main
scripts/ops/render_promote_main.sh --prod-url=https://protocol-bimy.onrender.com

# if prod did not update: check the service and read build logs
scripts/ops/render_deploy.sh status
scripts/ops/render_deploy.sh logs --limit=200
```

## 9) Smoke-check структуры скриптов

После `pull` на любом ПК можно быстро проверить, что wrapper-структура не сломана:

```bash
bash scripts/ops/smoke_repo_layout.sh
```
