# Ежедневный чеклист для 2 компьютеров

Короткая памятка для безопасной работы и деплоя в проекте Protocol.

## 1) Старт сессии

```bash
git checkout codex/main-sync
scripts/ops/git_safe_pull.sh
```

## 2) Работа и локальная проверка

- Внести изменения в рабочей ветке `codex/main-sync`.
- Прогнать релевантные тесты по измененным модулям.

## 3) Фиксация изменений

- Если изменение осмысленное - обновить `BUILD_VERSION` в `rag_server.py`.
- Сделать commit и push в рабочую ветку.

```bash
git add <files>
git commit -m "your message"
git push origin codex/main-sync
```

## 4) Промо в прод-ветку Render

Рекомендуемая единая команда (из любой рабочей ветки):

```bash
scripts/ops/deploy_promote_main_after_push.sh --prod-url=https://protocol-bimy.onrender.com
```

Она сама:
- пушит текущую ветку в origin;
- fast-forward продвигает текущий `HEAD` в `origin/main`;
- ждёт обновления `/api/version` на Render.

```bash
scripts/ops/render_promote_main.sh --prod-url=https://protocol-bimy.onrender.com
```

Важно: аргументы передавать как `--key=value` (а не через пробел).

## 5) Контроль деплоя

- Проверить версию на сервере:
  - [https://protocol-bimy.onrender.com/api/version](https://protocol-bimy.onrender.com/api/version)
- Версия должна совпадать с локальным `BUILD_VERSION`.

## Быстрая диагностика, если Render взял старый коммит

Проверить, что `main` и рабочая ветка указывают на один SHA:

```bash
git fetch origin --prune
git rev-parse origin/main
git rev-parse origin/codex/main-sync
```

Если SHA отличаются, выполнить шаг 4 (promote) и дождаться обновления версии.
