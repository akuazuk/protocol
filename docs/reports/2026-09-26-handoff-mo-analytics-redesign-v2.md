# Handoff: МО Аналитика, редизайн v2 - волны A, T, E, B, C в проде; D merged, релиз 4 откачен

Дата: 2026-09-26 (вечер UTC)
План: `docs/plans/2026-09-26-mo-analytics-redesign-v2.md` (active; журнал релизов - §6b,
метрики «было / стало / цель» - §7).
Прод: GCE `https://protocol.kravira.by`, контейнер `protocol-web`, образ
`protocol-gcp-app:4745229c642d`, `/api/version` = `2026-09-26-152505Z-mo-wave-c-weighted-axes`,
`git_commit` = `4745229c`. Render не прод и не откат.

Директива владельца: «Все подтверждаю работай автономно и все реализуй по плану». Режим:
одна волна = один PR (Bugbot по diff до merge) = merge после зелёного CI = релиз
`deploy_to_gce.sh` = приёмка (version/health, проба таймингов на GCE, DOM-аудит,
сценарии) = запись в план. Откат - предыдущий образ по SHA; один откат (релиз 4, см. ниже).

## Сделано (факты)

| Волна | PR | merge SHA | В проде |
|--|--|--|--|
| план v2 | #295 | `31d0bc24` | docs |
| T инструменты | #297 | `727d886a` | 306e29ec |
| E шрифты | #298 | `77e22fcd` | 306e29ec |
| A быстрые дефекты | #296 | `f07d3d2a` | 306e29ec |
| perf ICD 1 (предфильтр лексикона) | #301 | `306e29ec` | релиз 1, 14:38 UTC |
| B скорость | #299 | `a87e7e4c` | релиз 2, 15:13 UTC |
| perf ICD 2 (индекс `ru_title`, блоб кандидатов) | #302 | `262c5758` | 4745229c |
| C данные с 1 января (runner, YTD, granularity) | #300 | `4745229c` | релиз 3, 15:55 UTC |
| T проба `sort_by=overall` | #303 | `fb80f2ce` | не требует релиза |
| handoff + журнал релизов | #304 | `4ba84da3` | docs |
| D умный поиск | #305 | `caa79964` | релиз 4 17:40 UTC `PUBLIC_OK`, **откачен 17:48** на `4745229c` |
| D perf: FTS5 вместо LIKE (fix отката) | #306 | - | PR открыт, CI + Bugbot; ветка `cursor/mo-search-perf-pc1` |

Приёмка релизов (все `PUBLIC_OK`, `/health/live` ok):

- 306e29ec: DOM-аудит 11 экранов + разбор на 1024/1440 - `closedDetails 0`, `navHidden 0`,
  `more false`, `rawBr 0`, `webfonts 5`; 6 woff2 отдаются 200.
- a87e7e4c: проба на GCE (`/tmp/probe_release2_a87e7e4c.json` на VM): 17 из 18 в порогах;
  `/cases` 0,44 с, `queue_only` 0,46, `finding_family=lab` 0,60, `/facets` 0,46 -> 6 мс,
  `/freshness` 0,22 -> 5 мс, `/score-dashboard` 0,98, `/reports` 36 мс, `engine:
  facets_sql_v1`. Единственная строка над порогом - дефект пробы (`sort_by=score`,
  UI такого не шлёт) - PR #303.
- 4745229c: `/meta` содержит `ytd` и `granularities`; `/timeseries?period=ytd` -> `month`,
  9 точек 2026-01..2026-09, 18 мс; кнопка «С начала года» в HTML прода; полоса пресетов
  без горизонтального overflow на 740-1440 (Playwright, mock API); YTD-тайминги на GCE:
  `/cases` 0,72 с, `/summary` 1,4 с, `/facets` 2,6 с (first), `/score-dashboard` 5,4 с,
  `/drugs-labs-kpis` 9,7 с - закрываются F1/F4/F5 (SQL-агрегаты вместо Python-очереди).
- caa79964 (релиз 4, **откачен**): функционально верно - `гипертония` за месяц 257
  (фраза 15 / синонимы 174 / коды по названию 68), `I10` 0,39 с, `/search/plan` работает;
  но текстовый поиск 11,5 с за месяц и 70 с за YTD (цель ≤ 1 с), первый запрос приёмки
  ушёл в таймаут 60 с. Откат вручную по runbook §5: `docker inspect` -> тот же набор
  `-v/-e`, образ `protocol-gcp-app:4745229c642d`, `GIT_COMMIT_SHA` предыдущего релиза
  (первый запуск скопировал `GIT_COMMIT_SHA` нового релиза - `/api/version` показывал
  чужой SHA, перезапущено с правильным). Проверено: `version` = `…mo-wave-c-weighted-axes`,
  `git_commit` = `4745229c`, `/health/live` ok.

Причина и исправление (PR #306): LIKE с цепочками REPLACE × 3 регистра × до 12 синонимов
повторялись в WHERE, ORDER BY, COUNT и GROUP BY чипов - четыре прохода по 121 тыс. строк.
Теперь FTS5 `fact_mo_case_search` (триггеры на `fact_mo_case`, пересборка при расхождении,
создаётся при первом поиске после релиза), коды - `GLOB`, название/врач - подзапросы по
dim-таблицам. Синтетика 120 тыс. строк: 0,5-1,1 с на запрос. FTS5 есть в контейнере (3.46)
и в `/opt/protocol/venv-mis` (3.40) - триггеры безопасны для ночного конвейера.

Прочее, сделанное на GCE руками координатора (не в git):

- `night_kp_sync.sh` падал с `PermissionError` с 2026-08-26 - `chown` каталога корпуса,
  сверка прошла (`changed=104`). Rceth: периодического re-crawl никогда не было
  (только watchdog resume) - предложение: недельный cron, решение владельца.
- Очередь «Проанализировать» не обрабатывалась (нет воркера на хосте) - установлен
  `deploy/gcp-app/mo-ingest-queue.service` (systemd `--loop`), включён 12:58 UTC.
- `deploy/gcp-app/mo_backfill_range.sh` установлен в `/opt/protocol/deploy/gcp-app/`
  (owner `pavel`, 755). Тестовый день `2026-06-25` прогнан трижды: полный 58 мин
  (export 1, скоринг 28 при `--workers 2`, recompute 27+); после #301 recompute 15 мин;
  после #302 (релиз 4745229c) recompute 3,5 мин. Итог на день: ~33 мин, из них ~28 -
  скоринг под GIL.
- Профилирование: `py-spy` в `/tmp/pyspy-venv` на хосте (root):
  `sudo /tmp/pyspy-venv/bin/py-spy dump --pid $(pgrep -f "^python scripts/recompute_mo_days" | head -1)`.
- Проба таймингов лежит на VM в `/tmp/mo_api_latency_probe.py` (версия волны T); запуск:
  `export METHODIST_TOKEN="$(sudo docker exec protocol-web printenv METHODIST_TOKEN)";
  python3 /tmp/mo_api_latency_probe.py --base http://127.0.0.1:8000 --out /tmp/probe_<sha>.json`.
  Токен не печатать.

## Делается

- На GCE с 16:04 UTC: полный прогон `mo_backfill_range.sh 2026-01-01 2026-06-30` (от июня
  к январю, оценка ~4 суток; пауза 01:00-04:45 UTC и при живом `gce-night.lock`; держит
  `state/mo-backfill.lock`). Лог `/var/data/medical_exams/logs/gce-mo-backfill.log`,
  прогресс `state/mo_backfill_range.json`, мягкий стоп `touch state/mo_backfill_stop`.
  Деплой рестартует контейнер - runner повторяет день сам (3 попытки), но лучше не
  деплоить в середине скоринга дня.

## Не сделано

- Финальный `recompute` истории пациентов после всех месяцев; п. 5 волны C (лаборатория
  с 2025-12) - в G/F5.
- Приёмка волны D на проде (пороги §D плана, p95 ≤ 1 с) - после релиза 5 (#306).
  Скрипт приёмки: `/tmp/mo_wave/search_accept.py` на Mac, копия `/tmp/search_accept.py` на VM
  (`export METHODIST_TOKEN=…; python3 /tmp/search_accept.py --base http://127.0.0.1:8000`).
- Шаг 8 волны D (эмбеддинги «похожие») - после F. Ревью словаря `dx_aliases_ru.json`
  врачом - долг H1.
- Волны J, F1-F7, G, K, H1/I, H2/H3 - не начаты.
- Скриншоты прода с ФИО врачей в git не кладутся.

## Следующий безопасный шаг

1. Проверить прогон:

```bash
gcloud compute ssh protocol-app --zone=europe-central2-a --command='sudo tail -5 /var/data/medical_exams/logs/gce-mo-backfill.log; sudo cat /var/data/medical_exams/state/mo_backfill_range.json | tail -20'
```

2. Дождаться зелёного CI и Bugbot по #306, `gh pr merge 306 --squash --delete-branch`,
   затем релиз 5 из нового detached worktree на `origin/main` (`.env` симлинк), между днями
   backfill. После `PUBLIC_OK` - приёмка `/tmp/search_accept.py` на VM; если p95 > 1 с или
   ложные срабатывания - откат на `4745229c642d` тем же способом.
3. Прод сейчас на образе релиза 3 (`4745229c642d`), а `origin/main` уже содержит D
   (`caa79964`): не деплоить `origin/main` без #306.

## Базовые цифры утреннего аудита (PR #295, прод `8000354f`) - для сравнения

- overall_grade (сентябрь, clinical): good 389 (3,7%), fair 6151, poor 3846, important
  141, critical 0; zone1 ok 4,8%; zone2b ok 0 (weak 1946, bad 1983, na 6598); КП
  подобран 3929 / 10 527.
- Клинические МО с зонами только с 2026-07; январь-июнь ~77 тыс. документов без оценки;
  `secure_cases` с июня; лаборатория с 2025-12.
- `/cases` холодный 12,1 с, `sort_by` по баллу 4,5 с, очередь 5,7-6,1 с, `/freshness`
  3,6-3,9 с (все закрыты волной B, см. §7 плана).
- Поиск: `гипертония` 0 при `гипертензия` 157 и `I10` 168; опечатка 0; `близорукость` 0
  (волна D).
- Разбор: 21 `details`, 20 закрыты; меню: 4 пункта в «Ещё»; шрифт без webfont (закрыто A/E).
- Скриншоты аудита были в `/tmp/mo_audit` - в git не кладём (ФИО врачей).

## Запреты и особенности

- Деплой только `bash deploy/gcp-app/deploy_to_gce.sh` из detached worktree на
  `origin/main` с `.env` симлинком из основного checkout
  (`ln -s /Users/pavelkuzauka/Cursor_Folders/Protocol/.env .env`), иначе «need GOOGLE_API_KEY».
- Не деплоить 01:00-04:30 UTC и пока Rceth `running`; деплой рестартует контейнер -
  runner backfill повторяет день сам (3 попытки), но лучше деплоить между днями.
- Основной checkout `/Users/pavelkuzauka/Cursor_Folders/Protocol` на `main` грязный
  и отстаёт - не чинить pull/reset, работать в worktree.
- Заголовок для API МО - `X-Methodist-Token`, не Bearer.
- Открытый PR #113 показывается в `pr_dashboard` как HARD overlap по `rag_server.py`
  - это только строка `BUILD_VERSION`, зомби-PR старше 14 дней.

## Нельзя трогать параллельно

`clinical_knowledge/mo_backend.py` (`build_cases`, `_cases_sql_pageable`, `_build_facets_uncached`,
`build_timeseries`), `clinical_knowledge/mo_metrics.py`, `icd_mkb.py` (лексикон,
`ru_title`), `frontend/web/shared/mo-app.js`, `frontend/web/methodist/mis-kz-quality.html`,
`deploy/gcp-app/mo_backfill_range.sh`, `docs/plans/2026-09-26-mo-analytics-redesign-v2.md`.
