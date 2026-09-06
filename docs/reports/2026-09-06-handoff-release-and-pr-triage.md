# Handoff: релиз аудита на прод, разбор зависших PR, обязательные проверки

Дата: 2026-09-06. Предыдущий: `2026-09-05-handoff-production-readiness-audit.md`.

## Контекст в одну строку

Аудит из #192 доехал до прода. `protocol.kravira.by` работает на
`a592d588`, версия `2026-09-06-073651Z-deploy-lock-allowlist`.

## Координаты

| | |
|--|--|
| Repo | `github.com/akuazuk/protocol` |
| Прод | GCE `protocol-app` (`europe-central2-a`), `https://protocol.kravira.by` |
| Прод SHA | `a592d588fdd7eb428161024ad13e4e3948bb3754` |
| Прод `BUILD_VERSION` | `2026-09-06-073651Z-deploy-lock-allowlist` |
| Образ | `protocol-gcp-app:a592d588fdd7` (тег по SHA) |
| Релизный checkout | `/private/tmp/release-main` (detached на `origin/main`) |

## Сделано

### Обязательные проверки на `main`

Было две (`lint-and-test`, `manifest-mode`), стало шесть: добавлены
`docker-images`, `security-scan`, `e2e`, `hygiene`.

Включён `allow_auto_merge` и `allow_update_branch`: `gh pr merge N --squash
--auto` доводит PR до merge без сиделки, `gh pr update-branch N --rebase`
снимает состояние `BEHIND`.

**«Require review from Code Owners» не включён и включать пока нельзя.**
Коллаборатор один (`akuazuk`, admin), `enforce_admins: true`, самоодобрение
GitHub запрещает. Требование стало бы неисполнимым: смержить нельзя ничего.
Включать вместе со вторым ревьюером.

### Разбор зависших PR

| PR | Итог |
|---|---|
| #148 | **merged** `41dd6bb9` - фикс не был в `main` 22 дня |
| #158 | **merged** `59c030cb` - MIS SQL только с GCE |
| #113 | оставлен открытым, решение по методике за владельцем |
| #177, #168, #120, #97, #77 | закрыты с объяснением в каждом |

Пятёрка закрыта потому, что правила `next-chat-handoff.mdc` предписывали читать
августовский handoff как **актуальное состояние**, и merge вернул бы устаревшие
указания в документ, который агент читает как инструкцию. Само правило
исправлено: теперь оно указывает на последний по дате файл.

### Исходники корпуса и провенанс (#193, `bf42e40c`)

29 файлов, 15.1 МБ. 27 PDF редакций 2026 года лежали незакоммиченными с
30.05.2026. `sha256` всех 27 сверены с манифестом перед коммитом.

**`minzdrav_protocols/_manifest.jsonl` - тот манифест происхождения, который
аудит числил отсутствующим** (открытый пункт 2). 475 строк, в каждой `url` на
`minzdrav.gov.by`, `sha256`, `bytes`, `downloaded_utc`, `http_status`.
Заполнено 100%. Осталось связать его с путями чанков, чтобы цепочка
`источник -> PDF -> чанк -> ответ врачу` стала сквозной.

### Hotfix релиза (#204, `a592d588`)

Первый заход деплоя упал на сборке образа:

```
failed to compute cache key: "/requirements-rag.lock": not found
```

Деплой отправляет на VM не рабочее дерево, а `git archive` со списком путей.
`requirements-rag.lock` появился в Dockerfile вместе с `--require-hashes`, но в
список не попал. Джоба `docker-images` в CI это не видит: она собирает образ из
**полного** репозитория. Два разных build-контекста, проверялся один.

Прод не пострадал - сборка идёт до подмены контейнера. Добавлен
`tests/test_deploy_allowlist.py`: сверяет источники всех `COPY` в Dockerfile с
allowlist деплоя. Проверено мутацией: без пути в списке падают два теста.

## Smoke на проде (пройден)

- `/api/version`: `version` и `git_commit` совпали с релизным коммитом
- `/health/live`: 200
- Заголовки из аудита на месте: `content-security-policy`,
  `strict-transport-security` (`max-age=31536000; includeSubDomains; preload`),
  `x-frame-options: SAMEORIGIN`, `x-content-type-options: nosniff`,
  `referrer-policy: strict-origin-when-cross-origin`
- Порт `8000` снаружи недоступен
- `GET /` - 200, 1.2 МБ разметки
- `/api/search/run` возвращает реальные протоколы с обоснованием совпадения и
  навигацией по разделам
- В прод-манифесте 58 протоколов редакций 2026 года, включая ВРТ (пост. №29)

### Про `chunks_loaded: 0` в `/api/corpus-stats`

Это **штатно**, не регрессия. В контейнере `RAG_STARTUP_MODE=manifest`,
`RAG_LAZY_CHUNK_STORE=1`, `RAG_LAZY_RETRIEVE=1`: чанки не держатся в памяти, а
читаются по манифесту. На диске 10 частей, 417 МБ; манифест - 563 записи, 562
протокола. Не принимать этот ноль за пустой корпус.

## Известные проблемы, не закрытые в этой сессии

1. **Два деплоя на одной машине сталкиваются.** `deploy_to_gce.sh` пишет
   секреты в общий `/tmp/protocol-web-sm`, а python-шаг делает `rmtree` перед
   записью. Параллельный запуск удаляет файлы из-под первого:
   `Failed to read file [/tmp/protocol-web-sm/telegram-bot-token]`. Нужен
   каталог с pid или flock.
2. **Версии в Secret Manager растут на каждый деплой**, даже когда значения не
   менялись (`google-api-key` уже в районе 45). Нужна проверка «значение
   совпало - не добавлять версию».
3. **`HEAD /` отдаёт 404**, `GET /` - 200. Внешние мониторинги, которые
   опрашивают HEAD, будут видеть падение.
4. **10 PR от Dependabot** (#194-#203) - следствие `dependabot.yml` из #192.
   Четыре поднимают Python с 3.11 до 3.14 в Dockerfile'ах: не для автомержа.
5. **11 worktree в `/private/tmp`**, часть на смерженных ветках. Путают
   дашборд и агентов.
6. **`git_task_start.sh` печатает в подсказке Render-URL** как прод.
7. `strict: true` плюс шесть проверок = каждый merge отправляет соседние PR на
   новый круг CI (~16-20 мин). Для очереди из трёх PR это заметно. Merge queue
   решил бы, но требует отдельного решения.

## Следующая безопасная команда

```bash
python3 scripts/ops/pr_dashboard.py
```

## Чего не трогать параллельно

- `deploy/gcp-app/deploy_to_gce.sh` - до починки гонки по `/tmp`
- `docs/plans/README.md`, `eval/mo_score_calibration/` - держит открытый #113
- `requirements*.txt`, Dockerfile'ы, `.github/workflows/` - держат Dependabot PR
- Деплой запускает только один координатор: `/tmp` общий, гонка реальна
