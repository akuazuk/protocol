# Handoff: МО Аналитика, редизайн v2 - волны A-F5 в проде (релизы 1-15); дальше F6

Дата: 2026-09-27 (~12:45 UTC)
План: `docs/plans/2026-09-26-mo-analytics-redesign-v2.md` (active; журнал релизов - §6b,
метрики «было / стало / цель» - §7).
Карта проекта: `docs/reports/2026-09-27-mo-analytics-redesign-project-map.md`.
Прод: GCE `https://protocol.kravira.by`, контейнер `protocol-web`, образ
`protocol-gcp-app:5bee2668f30f`, `/api/version` = `2026-09-27-121936Z-labs-dash-f5-e2e`,
`git_commit` = `5bee2668` (релиз 15, 12:38 UTC 27.09). Render не прод и не откат. Предыдущий
образ для отката - `protocol-gcp-app:9927e256fed0` (релиз 14).

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
| D perf: FTS5 вместо LIKE (fix отката) | #306 | `2ca743fb` | релиз 5 19:03 UTC, приёмка частично (p95 4,35 с) |
| D perf 2: чип врача, один GROUP BY, один MATCH на синонимы, подфраза ОРВИ, прогрев индекса | #307 | `1ec1b39d` | релиз 6 20:02 UTC, 5/5 порогов, p95 2,28 с |
| D perf 3: название МКБ в FTS (схема v2), обратный индекс стемм, коды через `dim_diagnosis`, JOIN по надобности, `warm_caches` | #308 | `bc7a596a` | релиз 7 21:01 UTC, **приёмка §D пройдена**: p50 154 мс, p95 487 мс |
| D: названия для пустых `dim_diagnosis` из справочника | #309 | `51a031b1` | релиз 8 04:06 UTC 27.09 (попытка 1 в 21:37 - авто-откат: публичный `/api/version` не ответил за 15 с под скорингом backfill) |
| D close-out (план, handoff) | #310 | `e6303c72` | docs |
| J контракт фильтров, legacy-колонки, CRM-статус в SQL, hidden-хосты | #311 | `53791c4c` | релиз 9 05:33 UTC 27.09, **приёмка §J пройдена** (см. Статус J в плане) |
| J фасеты при deep-link (`ensureFacets`), приёмка J, план/handoff | #312 | `4d2a69d6` | релиз 10 06:44 UTC 27.09, приёмка пройдена: меню фасетов при `?page=documents` заполнены (врачи 100, специальности 23, филиалы 3, статусы 1) |
| F1 Обзор O1-O8 одним `/overview-dashboard` | #314 | `9174949f` | релиз 11 08:31 UTC 27.09, **приёмка §F1 пройдена** (месяц 1,49 с / YTD кэш 14 мс, 8 карточек, drill=cases) |
| F2 сводка «Найти МО» `/cases/summary` | #316 | `c52b9038` | релиз 12 09:32 UTC 27.09, **приёмка §F2 пройдена** (n=10 832 = `/cases`, Терапевт 2067, good 408, месяц тёплый 9-16 мс) |
| F3 Врачи D1-D5 | #318 | `70adb73d` | код в `a7279024` |
| F3 + природные цвета KPI | #319 | `a7279024` | релиз 13 11:06 UTC 27.09, **приёмка §F3 пройдена** (ранг 80, heatmap 12×3, scatter 80, фильтр врача, тёплый 10-79 мс; KPI вес 640 мох/роза) |
| F4 Лекарства M1-M5 | #321 | `9927e256` | релиз 14 11:47 UTC 27.09, **приёмка §F4 пройдена** (типы 168/92/7326/0, specialty 12, drugs 20, trend 4 нед., пары скрыты, DOM 5 KPI + 4 canvas) |
| F5 Анализы L1-L5 | #323 | `5bee2668` | релиз 15 12:38 UTC 27.09, **приёмка §F5 пройдена** (окно 1730/9102/275, тесты 15, specialty 12, coverage 10 мес., DOM 6 KPI + 5 canvas) |

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
dim-таблицам. FTS5 есть в контейнере (3.46) и в `/opt/protocol/venv-mis` (3.40) - триггеры
безопасны для ночного конвейера.

Релизы 5-7 (поиск, приёмка `/tmp/search_accept.py` на VM, 5 золотых + 20 контрольных
+ 40 замеров задержки):

- 2ca743fb (релиз 5): 4 из 5 порогов, p95 4,35 с - чип «врач/ID» с `CAST(visit_id) LIKE`
  по всем строкам, «острая респираторная» 30 (не ключ группы ОРВИ). Не откатывался.
- 1ec1b39d (релиз 6): 5 из 5, «острая респираторная» 11 622, p95 2,28 с. Профиль на GCE
  (`cProfile` + `set_trace_callback` в контейнере): 1,1 с Python `codes_for_phrase`,
  0,7-1,1 с SQL - подзапросы `dim_diagnosis` с REPLACE-цепочками в WHERE и ранге,
  `LEFT JOIN` к справочникам на каждую строку, `GLOB` по кодам на каждую строку.
- bc7a596a (релиз 7): 5 из 5, 0 ложных, p50 154 мс, p95 487 мс, max 684 мс («СД 2»,
  цифра в запросе -> `CAST(visit_id) LIKE`). Индекс v2 пересобран при старте за ~7 с
  (в логе `MO search caches prewarmed: icd_titles 15616, stems 5973, aliases 950`),
  121 519 строк = фактов, `fact_mo_case_search_meta`: `schema_version 2`,
  `dim_fingerprint 2441:0`. Отпечаток `…:0` = у всех кодов `dim_diagnosis` пустое
  название - исправление в #309 (2362 из 2441 кодов получат название из справочника).
- 53791c4c (релиз 9, J): API за 01-25.09 - каждый чип UI меняет `total` (10 527 базово;
  `overall_grade=critical|important|poor` 3987, `kp_status=unmatched` 6598,
  `history_tier=first_contact` 6709, `queue_only` 6627, `q=гипертония` 257);
  `crm_statuses=in_review` за 07-09.2026 - 11 строк, facets `[new 40 496, in_review 11]`;
  `critical` 0 - в данных нет. DOM-аудит: `closedDetails 0`, `navHidden 0`, `rawBr 0`,
  ошибок JS 0; заголовки 15 / 18 / «Оценка» первой в таблице дня; `#queue-critical-only`
  нет; hidden-хостов нет; панель «Фильтры» внутри контента на 1280 и 1024. Результат
  `/tmp/mo_wave/dom_audit_53791c4c.json` (сравнение: `--compare dom_audit_306e29ec.json
  dom_audit_53791c4c.json`). Скрипты приёмки: `/tmp/j_accept.py` на VM,
  `/tmp/mo_wave/j_dom_check.mjs`, `j_panel_check.mjs` на Mac (запуск из worktree с
  симлинком `node_modules`, токен из env).
- Для замеров на копии склада без риска для прода: `cp mo_analytics.sqlite /tmp/mo_exp.sqlite`
  внутри контейнера + `PYTHONPATH=/tmp/newcode:/app MO_ANALYTICS_DB=/tmp/mo_exp.sqlite`
  (в `/tmp/newcode` - новый `clinical_knowledge`, `data` симлинком на `/app/data`).
  Скрипты на Mac: `/tmp/mo_wave/bench_new.py`, `profile_sql3.py`, `profile_ytd.py`.

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
- Волна D закрыта (релиз 8: `dim_fingerprint 2443:95566`, 79 кодов без названия -
  нестандартные `A05.05`…, «гипертензивная болезнь» 120 по фразе, p95 381 мс).
  Скрипт приёмки: `/tmp/search_accept.py` на VM
  (`export METHODIST_TOKEN=…; python3 /tmp/search_accept.py --base http://127.0.0.1:8000`).
- «СД 2»: 3 из 10 строк выборки эвристика скрипта не объясняет (она не видит
  `diagnosis_text`); движок даёт ранг 2 по целому слову «СД» - проверить методисту вручную.
- Шаг 8 волны D (эмбеддинги «похожие») - после F. Ревью словаря `dx_aliases_ru.json`
  врачом - долг H1.
- Волна J закрыта релизом 9 (детали и цифры приёмки - Статус J в плане). Долги J -> H2:
  «две "критично"» (плитка P0/P1 против чипа grade), чипы `worst_severity` и кольцо
  приоритетов с P-кодами внутри, `sort_by=priority` в селекте.
- Найдено при приёмке J на проде: меню фасетов (врачи, филиалы, специальности, статус
  разбора) пусты при deep-link `?page=documents` / `?page=queue` - `/cases` не отдаёт
  `facets`, `/facets` запрашивал только «Обзор» (дефект с `cbeb805c`). Фикс -
  `ensureFacets()` в `loadPage` и `beginFilterDraft`, PR `cursor/mo-redesign-j-facets-deeplink-pc1`.
  Релиз 10 принят на проде 06:44 UTC: `?page=documents` -> «Фильтры» -> меню заполнены
  (скрипт `/tmp/mo_wave/j_panel_check.mjs` на Mac). **Волна J закрыта.**
- `icd_chapter` пуст у 23% случаев сентября (2472 из ~10,5 тыс.) - в аудит данных I.
- Волна F1 закрыта релизом 11 (цифры - Статус F1 в плане). Холодный YTD 8,8 с закрывается кэшем 120 с (повтор 14 мс); отдельно не ускоряли.
- Волна F2 закрыта релизом 12 (цифры - Статус F2 в плане). Холодный месяц 2,87 с / тёплый 9-16 мс; YTD 2,56 с. Overflow широкой таблицы «Найти МО» - долг J/F (15 колонок), не сводка.
- Волна F3 закрыта релизом 13. Волна F4 закрыта релизом 14. Волна F5 закрыта релизом 15. F6-F7, G, K, H1/I, H2/H3 не начаты.
- Подробная карта: `docs/reports/2026-09-27-mo-analytics-redesign-project-map.md`.
- Скриншоты прода с ФИО врачей в git не кладутся.

## Следующий безопасный шаг

1. Проверить прогон:

```bash
gcloud compute ssh protocol-app --zone=europe-central2-a --command='sudo tail -5 /var/data/medical_exams/logs/gce-mo-backfill.log; sudo cat /var/data/medical_exams/state/mo_backfill_range.json | tail -20'
```

2. Волна F6 (Поиск МИС + Очередь, в том числе overflow широких таблиц) по плану §5.6/§5.7 -
   один PR от свежего `origin/main`, Bugbot, merge, релиз 16 между днями backfill.
3. Перед любым деплоем проверить `date -u`: не 01:00-04:30 UTC. Релиз 8 попал в окно
   (04:01) из-за сна Mac между командами; ночной конвейер уже завершился (02:15), вреда нет.

## Как деплоить при живом backfill

Попытка 1 релиза 8 откатилась автоматически: контейнер стартовал одновременно со
скорингом backfill (2 воркера `docker exec` внутри того же контейнера на 2 vCPU) и
прогревом (заливка названий + пересборка FTS 7 с) - публичный `/api/version` не ответил за
15 с, скрипт счёл версию неверной. Порядок, который сработал:

```bash
# 1. мягкий стоп после текущего дня (флаг runner снимает сам)
gcloud compute ssh protocol-app --zone=europe-central2-a --command='sudo -u pavel touch /var/data/medical_exams/state/mo_backfill_stop'
# 2. дождаться "stop flag present - exiting" в логе и отсутствия процесса
#    (pgrep -f "bash /opt/protocol/deploy/gcp-app/mo_backfill_range.sh"; просто "mo_backfill_range" ловит сам ssh)
# 3. deploy_to_gce.sh
# 4. перезапуск runner (resume-safe: дни с маркером state/mo_backfill_done_<день> пропускаются)
gcloud compute ssh protocol-app --zone=europe-central2-a --command='sudo -u pavel bash -c "cd /opt/protocol && nohup bash /opt/protocol/deploy/gcp-app/mo_backfill_range.sh 2026-01-01 2026-06-30 >> /var/data/medical_exams/logs/gce-mo-backfill.log 2>&1 &"'
```

Шаг 4 держит ssh-сессию открытой (nohup наследует stdout) - запускать в фоне/с таймаутом.
Runner перезапущен 04:08 UTC 27.09, остановлен мягко 05:22 (после 06-02) для релиза 9 и
перезапущен 05:34; остановлен мягко 06:37 (после 05-28) для релиза 10 и перезапущен 06:44;
остановлен мягко после 05-21 для релиза 11, перезапущен после приёмки (с 05-19);
остановлен мягко после 05-17 для релиза 12, перезапущен после приёмки (с 05-16). День
`2026-06-22` в состоянии `score_failed` (без маркера) - runner его повторит. Длинные ssh-циклы ожидания (> ~15 мин)
рвутся с exit 255 - опрашивать короткими отдельными ssh.

Грабли этой сессии, чтобы не повторять: `pytest … | tail && git commit` глотает код
выхода pytest (в #311 ушёл коммит с красным тестом, починен следующим) - проверять
`${PIPESTATUS[0]}` или не использовать пайп; `scripts/normalize_ui_dashes.py` игнорирует
пути и переписывает весь репозиторий - лишние файлы возвращать `git checkout --`;
`gh pr create --body "$(cat <<EOF…)"` дописывает vendor-атрибуцию - только `--body-file`.

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
