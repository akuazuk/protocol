# Handoff: UX rebuild МО Аналитики, волны W0-W7

Дата: 2026-09-19
Репозиторий: `https://github.com/akuazuk/protocol.git`
Прод: GCE `https://protocol.kravira.by` (единственный контур)

## Репозиторий и ветки

| | |
|--|--|
| Worktree волн | `/private/tmp/protocol-task-mo-find-cases-wN-1` |
| Close-out | `/private/tmp/protocol-task-mo-ux-plan-close-1`, ветка `cursor/mo-ux-plan-close-pc1` |
| Base / HEAD прод | `origin/main` = `58c956d4` |
| Не трогать | грязный Cursor checkout `/Users/pavelkuzauka/Cursor_Folders/Protocol` на `main` |

## Сделано (в проде)

| Волна | PR | merge SHA | `BUILD_VERSION` | GCE |
|--|--|--|--|--|
| W0 фильтры | #251 | `20a3f822` | | PUBLIC_OK |
| W1 stale list | #252 | `582329f6` | `2026-09-19-150608Z-mo-list-stale` | PUBLIC_OK |
| W2 SQL LIMIT | #253 | `82b32af7` | `2026-09-19-154804Z-mo-sql-limit` | PUBLIC_OK |
| W3 полоса поиска | #254 | `9de0be1d` | `2026-09-19-161513Z-mo-find-bar` | PUBLIC_OK |
| W0.7 queue_band | #255 | `186719fa` | | PUBLIC_OK |
| W4 меню 5+Ещё | #256 | `264b2a2b` | `2026-09-19-172903Z-mo-nav-more` | PUBLIC_OK |
| W5 инспектор | #257 | `e5cbadab` | `2026-09-19-183601Z-mo-inspector` | PUBLIC_OK |
| W6 линзы семьи | #258 | `8b4d0a82` | `2026-09-19-185950Z-mo-family-lenses` | PUBLIC_OK |
| W7 полировка | #259 | `58c956d4` | `2026-09-19-192135Z-mo-table-polish` | PUBLIC_OK |

Smoke после W7:

- `/api/version` = `2026-09-19-192135Z-mo-table-polish`, `git_commit` = `58c956d4…`
- `/health/live` ok
- «Найти МО»: 50 строк, пресеты Работа/Проверка в «Колонки»
- `/cases?period=month` total=7809; `overall_grade=good` 319; `queue_band=critical` 13
- Отчёты: 120 дневных файлов (empty-state в коде, на проде список не пустой)

## Не сделано / хвосты

- p95 `/cases` page=1 не < 1.5 с (~3-8 с на GCE PD)
- `overall_grade` всё ещё Python-скан
- W7 п.3: легенда колец Обзора = шкала оценки, клик сегмента
- W7 п.5: чип корпуса КП 478 vs диск после деплоя
- Плитка Обзора «Критично в очереди» на зерне Месяц может показывать 0 при `/cases?queue_band=critical` = 13 - не чинить попутно

## Тесты

W7: `tests/test_mo_polish_w7.py` + структура W4-W6, CI PR #259 зелёный (lint-and-test, e2e, hygiene).

## Запреты

- `SYNC_PROTOCOL_CORPUS=1`
- Render как прод или откат
- Порт 8000 наружу
- Чинить грязный Cursor `main`
- Gemini/MIS SQL с Mac
- `rag_server.py` и `docs/plans/README.md` заняты открытыми #186 и #113 - close-out их не трогает, `BUILD_VERSION` не поднимался

## Одна следующая команда

Новую UX-волну не начинать без выбора хвоста владельцем. Кандидат - индексы склада:

```bash
scripts/ops/git_task_start.sh mo-cases-sql-indexes --pc=1 \
  --branch=cursor/mo-cases-sql-indexes-pc1
```

Деплой координатор: `SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh` только если HEAD = `origin/main`, из worktree с `.env`.
