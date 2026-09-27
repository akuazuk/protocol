# Карта проекта: МО Аналитика, редизайн v2

Дата: 2026-09-27 (~11:00 UTC).
Для следующего агента: читать этот файл вместо повторного обхода всего репозитория.
Канон плана: `docs/plans/2026-09-26-mo-analytics-redesign-v2.md`.
Журнал релизов: там же, §6b. Метрики: §7.
Предыдущий живой handoff: `docs/reports/2026-09-26-handoff-mo-analytics-redesign-v2.md`.

Не читать как текущее состояние: `2026-08-20-handoff-next-chat.md` и августовские handoff
(с тех пор прод только GCE, Render отключён).

---

## 0. Очередь: что закончено, что нет

**План целиком не закончен.** Закрыты волны A, T, E, B, C, D, J, F1, F2 (в проде,
релизы 1-12). F3 смержена в `origin/main` (`70adb73d`, PR #318), **ещё не задеплоена**.
F4-F7, G, K, H1/I, H2/H3 не начаты.

| Волна | Смысл | Где сейчас |
|--|--|--|
| A | Убрать «Ещё» и свёртки, все экраны видны | прод, в релизе 1 (`306e29ec`) |
| T | probe / warehouse profile / DOM-audit | в репо, гоняются с Mac и GCE |
| E | webfont Onest + Golos + JetBrains Mono | прод, 6 woff2 = 200 |
| B | скорость `/cases` `/facets` `/freshness`, ingest-воркер | прод, релиз 2 |
| C | период YTD, кнопка «С начала года», backfill янв-июнь | код в проде; данные ещё считаются |
| D | умный поиск FTS5 + синонимы + опечатки + названия МКБ | прод, релизы 5-8 (релиз 4 откачен) |
| J | контракт фильтров, legacy-колонки, CRM в SQL, фасеты deep-link | прод, релизы 9-10 |
| F1 | Обзор O1-O8 одним `/overview-dashboard` | прод, релиз 11 `9174949f` |
| F2 | сводка «Найти МО» `/cases/summary` | прод, релиз 12 `c52b9038` |
| F3 | Врачи D1-D5 `/doctors-dashboard` | `origin/main` `70adb73d`, **не в проде** |
| F4 | Лекарства M1-M5 | не начато |
| F5 | Анализы L1-L5 | не начато |
| F6 | Поиск МИС + Очередь | не начато (долг: overflow широких таблиц) |
| F7 | Отчёты / Протоколы МЗ / Rceth | не начато |
| G | разбор случая без свёртков | не начато |
| K | пациент во времени | не начато |
| H1/I | разметка findings + аудит данных | не начато |
| H2/H3 | шкала, две «критично», перекалибровка | не начато |

Прод на момент этой карты: `https://protocol.kravira.by`
`/api/version` = `2026-09-27-091249Z-mo-redesign-f2-summary`,
`git_commit` = `c52b90389f08` (релиз 12). Образ `protocol-gcp-app:c52b90389f08`.
Откат: `protocol-gcp-app:9174949f9097` (релиз 11). Docs-only `277d0cf2` на main
не деплоился (как F1-паттерн: docs отдельно).

Открытый продуктовый PR после merge #318: нет. Зомби: #261 (план workspace),
#113 (calibration, HARD overlap только `BUILD_VERSION`), #186 (статья РЗ),
Dependabot #194-#203, #248. Их не мержить попутно с редизайном.

---

## 1. Что это за продукт

МО Аналитика - кабинет методиста качества ОДО «Медицинский центр «Кравира»».
Считает клинические консультативные заключения (КЗ / type=9 в МИС) по трём зонам
(оформление, диагноз, план по КП) и пятиступенчатой оценке
(Критично / Важно / Слабо / С замечанием / Хорошо / Нет оценки).
Методист ищет плохие и хорошие МО, разбирает случай, сохраняет решение.

Вход: `https://protocol.kravira.by/methodist/mis-kz-quality.html`
(или тот же origin, страница кабинета). Заголовок `X-Methodist-Token`.
Токен в `.env` основного checkout и в `localStorage protocol_methodist_token`.
Не печатать токен и PHI (ФИО, полные тексты КЗ) в git / PR / чат.

Канон языка и зон: `docs/plans/2026-08-08-mo-analytics-ui-target-v2.md`.
Шкала: `docs/plans/2026-08-20-mo-grade-ladder-v1.md`.
Этот редизайн: `docs/plans/2026-09-26-mo-analytics-redesign-v2.md` (active).

---

## 2. Где живёт прод

Единственный прод - GCE.

| | |
|--|--|
| UI | `https://protocol.kravira.by` |
| Проект / зона / VM | `protocol-home-e1` / `europe-central2-a` / `protocol-app` (e2-standard-2, 2 vCPU) |
| Контейнер | `protocol-web` за Caddy, слушает `127.0.0.1:8000` |
| Образ | `protocol-gcp-app:<sha12>` |
| Данные МО | `/var/data/medical_exams` на диске `protocol-data` |
| Склад | `/var/data/medical_exams/warehouse/mo_analytics.sqlite` (+ `mo_lab.sqlite`) |
| Тексты КЗ | `/var/data/medical_exams/secure_cases/YYYY/MM/` |
| Логи backfill | `/var/data/medical_exams/logs/gce-mo-backfill.log` |
| Состояние backfill | `/var/data/medical_exams/state/mo_backfill_*.json`, маркеры `mo_backfill_done_<день>` |
| Мягкий стоп | `sudo -u pavel touch /var/data/medical_exams/state/mo_backfill_stop` |
| Runbook | `docs/deploy/gce-production-runbook.md` |
| Деплой | `bash deploy/gcp-app/deploy_to_gce.sh` из detached worktree на **ровно** `origin/main` |

Render (`protocol-bimy.onrender.com`) приостановлен, отдаёт 503. Не прод и не откат.
Порт 8000 наружу закрыт (только 80/443). Gemini и SQL к МИС - только с GCE,
не с Mac (geo 400 / правило MIS).

Ночной конвейер: 01:00-04:30 UTC плюс `gce-night.lock`. В это окно не деплоить.
Пока Rceth `running` - не деплоить.

---

## 3. Как устроены данные

Источник истины по визиту - MariaDB МИС `kravira_mc.mis_protocol` (type=9 = КЗ).
Подключение только с GCE: `deploy/gcp-app/load_mis_env.sh` + Secret Manager
`kravira-db-password`. Поля `result` режутся по `::`, индекс N = слот схемы
(не порядковый номер). Клинический текст: слоты 3 жалобы, 10 анамнез, 4 статус,
11 обследования, 5 диагноз, 22 структурированный диагноз (`##` / `|`),
6/26 рекомендации. ФИО врача не в `result`, а в `mis_data.specialist_name`.

Ночной extract (cron 02:00 UTC) кладёт secure_cases и обновляет склад.
Склад `fact_mo_case` - витрина, не источник диагноза. Оценка пишется скорером
(`scripts/recompute_mo_days.py` + зональный движок в `clinical_knowledge/`).

Покрытие на утро 26.09 (аудит плана): ~121 тыс. строк за 2026, из них clinical
с зонами только июль-сентябрь (~24,5 тыс.). Январь-июнь в складе как consultation /
diagnostic без зон. Backfill `mo_backfill_range.sh 2026-01-01 2026-06-30` идёт
с июня к январю, ~35 мин/день, 2 воркера `docker exec` внутри того же контейнера.
День `2026-06-22` - `score_failed`, runner повторит (нет маркера done).
После F2 runner ушёл с 05-16; к 10:44 UTC 27.09 дошёл до 2026-05-12 (460/460).

Лаборатория: `mo_lab.sqlite` / `fact_mo_lab` с 2025-12 (~485 тыс. строк).
В primary-оценку пока не входит (долг H1 / F5).

Поиск: FTS5 таблица `fact_mo_case_search` схема v2, триггеры на `fact_mo_case`.
Отпечаток словаря `dim_diagnosis` после релиза 8: `2443:95566` (2364 кода с
названием из `icd10_ru_mkb10su.json`, 79 нестандартных пустые).
Прогрев: `warm_caches` в логе старта (`icd_titles`, `stems`, `aliases`).

Специальности в фасетах - должности (`Терапевт`), не профили (`Терапия`).
`icd_chapter` пуст у ~23% сентября - чип «Раздел МКБ» на них молчит (долг I).

---

## 4. Где лежит код (что трогать в какой волне)

Один PR - один экран / один модуль. Не мешать `frontend/web/` с `.github/` и
`tests/` в одном PR (CODEOWNERS уровни 3 и 4).

| Путь | Роль |
|--|--|
| `clinical_knowledge/mo_backend.py` | все сборщики витрин: `build_cases`, `build_cases_summary`, `build_overview_dashboard`, `build_doctors_dashboard`, facets, timeseries, `_warehouse_where`, `_cached_result`, `_sql_overall_grade_expr`, `_dim_joins_sql` |
| `clinical_knowledge/mo_overall_grade.py` | `compute_mo_overall_grade`; `na` если нет ни одной зоны и нет safety |
| `clinical_knowledge/mo_zone_scores.py` | зоны 1 / 2a / 2b |
| `clinical_knowledge/icd_mkb.py` | лексикон МКБ, `ru_title`, предфильтр |
| `rag_server.py` | FastAPI маршруты `/api/methodist/mo/*` + `BUILD_VERSION` |
| `frontend/web/methodist/mis-kz-quality.html` | разметка кабинета, лимит 480 строк |
| `frontend/web/shared/mo-app.js` | состояние, фильтры, рендер экранов, drill в «Найти МО» |
| `frontend/web/shared/mo-api.js` | `request()`, база `/api/methodist/mo`, fallback 404 |
| `frontend/web/shared/mo-charts.js` | обёртка ECharts: `MO.moChart`, `MO.moDonut` |
| `frontend/web/shared/mo-ui.css` | сетки экранов, KPI, таблицы |
| `frontend/web/shared/mo-tokens.css` | шрифты, цвета, `--num-*` для цифр дашборда |
| `frontend/web/shared/vendor/fonts/` | 6 woff2 (волна E) |
| `deploy/gcp-app/deploy_to_gce.sh` | единственный релиз |
| `deploy/gcp-app/mo_backfill_range.sh` | янв-июнь, resume-safe |
| `deploy/gcp-app/mo-ingest-queue.service` | кнопка «Проанализировать» (B5) |
| `scripts/ops/mo_api_latency_probe.py` | тайминги с GCE |
| `scripts/ops/mo_warehouse_profile.py` | профиль склада |
| `scripts/ops/mo_ui_dom_audit.js` | DOM-аудит |
| `scripts/ops/git_task_start.sh` | worktree от `origin/main` |
| `scripts/ops/bump_build_version.sh` | UTC-метка в `rag_server.py` |
| `tests/test_mo_*_f1.py` / `_f2.py` / `_f3.py` | контракты волн |
| `tests/test_mo_ui_phase2.py` | structure: хосты, CSS, без em-dash |
| `tests/test_mo_cases_query_contract.py` | каждый чип меняет `total` |

Кэш витрин: `_cached_result(name, params, ttl, producer)`. TTL: overview 120 с,
summary 60 с, doctors 90 с. Ключ включает штамп склада (`_warehouse_stamp`),
поэтому WAL-запись backfill сбрасывает кэш - первый YTD после скоринга снова
холодный.

Фильтры режутся одним `_warehouse_where` / `_sql_case_filter`. Любой новый
дашборд обязан брать тот же WHERE, иначе n разъедется с `/cases`.

Оценка в SQL: `_sql_overall_grade_expr`. Записанная скорером `overall_grade`
первична. `na` - нет зон и нет safety (раньше такие строки падали в `fair`).

---

## 5. API, которые уже работают

Префикс `/api/methodist/mo`. Заголовок `X-Methodist-Token`.

| Метод | Путь | Волна | Что отдаёт |
|--|--|--|--|
| GET | `/cases` | B, J, D | страница случаев, тот же WHERE |
| GET | `/cases/summary` | F2 | grades / specialties top-8 / weeks / search_plan, кэш 60 с |
| GET | `/overview-dashboard` | F1 | O1-O8 одним ответом, кэш 120 с |
| GET | `/doctors-dashboard` | F3 | D1-D5, кэш 90 с; **есть на main, нет на проде** |
| GET | `/score-dashboard` | старше F1 | fallback Обзора, если overview 404 |
| GET | `/facets` | B, J | врачи / филиалы / специальности / crm_statuses |
| GET | `/dimensions/doctors` | старше F3 | fallback страницы Врачи |
| GET | `/search/plan` | D | чипы фраза / синонимы / коды |
| GET | `/freshness` | B | свежесть склада |
| GET | `/meta` | C | `ytd`, granularities |
| GET | `/timeseries` | C | точки по grain |
| GET | `/drugs-labs-kpis` | старше F | семьи drug/lab; F4/F5 перепишут в дашборд |
| GET | `/health/live`, `/api/version` | всегда | smoke релиза |

Маршрут `/cases/summary` и `/doctors-dashboard` зарегистрированы **до**
`/cases/{case_id}`, иначе FastAPI съест `summary` как id.

Период: `_apply_request_period` + `_apply_score_eligible_default`.
Пресеты: yesterday / 7d / month / ytd / custom. Grain авто: день ≤62 сут.,
неделя ≤190, дальше месяц.

---

## 6. Экраны UI и как они устроены

Один HTML: `frontend/web/methodist/mis-kz-quality.html`. Страницы - секции
`.page` с `data-page`. Меню без «Ещё» (волна A). Состояние фильтров в
`mo-app.js` (`state.*`), URL через `filtersToSearchParams` / чипы шапки.

### Обзор (`#page-yesterday`, F1)

Сетка `.overview-grid`, 8 карточек. Данные: `GET /overview-dashboard`.
Fallback: `/score-dashboard` при 404. Клик по любой диаграмме - `applyDrill`
на «Найти МО» (оценка / специальность+неделя / finding / kp_status / зона).

- O1 лента оценок (кольца `moDonut({ compact })`, 150 px, общая легенда)
- O2 зоны с `prev_ok_pct` / `delta_ok_pct` (zone2b без КП - без дельты)
- O3 тренд + `trends_compare` (прошлый период выровнен по индексу)
- O4 heatmap специальность × неделя ISO `%W`, ячейки n<5 как `<5`, до 12×26
- O5 топ findings `passed=0`, подписи `mo_finding_labels_ru`
- O6 воронка КП `zone2b_kp_status`
- O7 мини-бары получено / ожидалось / допущено / оценено (`#yesterday-completeness`)
- O8 таблица дня 8 колонок, `table-dense`, 36 px, overflow 0 на 1280/1024

`details` на Обзоре нет. Тесты: `tests/test_mo_overview_dashboard_f1.py`.

Приёмка прод (месяц): n=10 832, good 408 / fair 6341 / poor 3938 / important 145 /
critical 0 / na 0; zone1 ok 4,8 (+0,5); 1,49 с. YTD холодный 8,8 с / кэш 14 мс,
n~98 856 (na 58 499 - янв-июнь без оценки).

### Найти МО (`#page-documents`, F2 + J)

Сверху `#cases-summary`: оценки (6 баров включая na), специальности топ-8,
спарклайн недель. Клик режет `overall_grade` / `specializations` / период.
Таблица 15 колонок без «Статус» / «Итог» / «Полнота» / «Надёжность»; есть МКБ,
История, КП. Пресеты колонок «Работа» / «Проверка». Chrome серверный: поиск → `q`,
«Только плохо» → `overall_grade=critical|important|poor`.

Приёмка: n сводки = `/cases`.total 10 832; Терапевт 2067=2067; good 408=408;
месяц 2,87 с холод / 9-16 мс тепло. Overflow 105@1280 / 307@1024 - широкая
таблица, не сводка (долг F6).

### Врачи (`#page-doctors`, F3 на main)

На проде ещё старый scatter в `<details>`. После деплоя `70adb73d`:
`.doctors-grid` с `#doctor-heatmap`, `#doctor-scatter`, `#doctor-trend`,
`#doctor-profile-radar`, `#doctor-profile-findings`, плюс старые
`#doctor-zone-chart` и таблица. Один запрос `/doctors-dashboard`.
Ранг: GROUP BY `c.doctor_key`, `enough` = n≥20. Выбранный врач: первый из
фильтра `doctors`, иначе первый enough. `specialty_median` считается в Python
(`statistics.median`), не AVG (фикс Bugbot). Fallback `/dimensions/doctors`
очищает графики (`renderDoctor*(null)`), иначе остаются чужие серии.
`enough` на fallback: `enough_data && !suppressed`.

Тесты: `tests/test_mo_doctors_dashboard_f3.py`. HTML ≤480 строк.

### Очередь (`#page-queue`)

18 колонок, «Оценка» вместо «Приоритет», CRM-статус подписан «Разбор».
`queue_only=1`. Кнопки «Только критические» нет - чип оценки «Критично».
F6 ещё не дала свои диаграммы.

### Лекарства / Анализы

Пока KPI-плитки `/drugs-labs-kpis` и таблицы findings. Дашборды M1-M5 / L1-L5
- волны F4 / F5.

### Поиск МИС, Отчёты, Протоколы МЗ, Rceth, Справка

Живые экраны без нового каталога диаграмм (F6/F7). Сверка КП МЗ починена
26.09 (`chown` корпуса, `changed=104`). Rceth без периодического re-crawl
(решение владельца). Справка = бывшие Настройки, зоны без v3/v4.

### Разбор случая (drawer)

`details[open]` после волны A (0 закрытых в DOM-аудите). Три колонки, липкие
якоря, спарклайны лаб, кнопки догрузки - волна G, не сделано.

---

## 7. Как сделана каждая закрытая волна (способ, не только факт)

### A - ничего не скрывать

Убраны пункты меню в «Ещё» и `hidden` на `.nav-button`. В разборе все
`<details>` открыты по умолчанию. Тест DOM-audit: `closedDetails 0`,
`navHidden 0`, `more false`. PR #296, влито в релиз 1.

### T - измерять до и после

`scripts/ops/mo_api_latency_probe.py` гоняется **из контейнера** на GCE
(токен из `docker exec protocol-web printenv METHODIST_TOKEN`, не печатать).
`mo_warehouse_profile.py` - распределения склада. `mo_ui_dom_audit.js` -
карточки / ECharts / overflow / шрифты / details / hidden nav, через CDP.
Сравнение двух JSON: `--compare old.json new.json`. PR #297, #303
(`sort_by=overall` вместо несуществующего `score` в пробе).

### E - шрифты

Самохостинг OFL: Onest (заголовки и KPI), Golos Text (UI/таблицы),
JetBrains Mono (коды). Файлы `frontend/web/shared/vendor/fonts/`.
`font-display: swap`, variable weight, кириллица отдельным unicode-range.
PR #298.

### B - скорость списка

`/cases` переведён на SQL-пейджирование и кэш. `/facets` - `facets_sql_v1`.
`/freshness` 0,22 с → 5 мс. `/cases` холодный 12,1 с → 0,69 / 0,44.
Воркер ingest: `mo-ingest-queue.service` на хосте (кнопка «Проанализировать»
больше не ждёт 02:00). PR #299, релиз 2.

### C - данные с 1 января

`/meta` отдаёт `ytd` и granularities. Кнопка «С начала года» в полосе пресетов.
`mo_backfill_range.sh` на GCE, от июня к январю, пауза 01:00-04:45 UTC и при
`gce-night.lock`. ICD perf 1-2 (#301, #302): предфильтр лексикона и индекс
`ru_title` - recompute дня 27 мин → 3,5 мин. PR #300, релиз 3.

### D - умный поиск

Первый PR #305 (LIKE + REPLACE по 121 тыс. строк) принят функционально и
**откачен вручную** за 11,5 с / 70 с YTD. Дальше FTS5 (#306-#308):
таблица `fact_mo_case_search` v2, MATCH на фразу и стеммы, коды через
`dim_diagnosis` + GLOB, чип врача без `CAST(visit_id) LIKE` на всех строках,
подфраза ОРВИ, `warm_caches`. Приёмка: 5/5 порогов, p50 154 мс, p95 487 мс.
Названия пустых `dim_diagnosis` - #309, релиз 8 (попытка 1 авто-откат:
публичный `/api/version` не ответил за 15 с под скорингом backfill).
Эмбеддинги «похожие» - шаг 8, после F. Словарь `dx_aliases_ru.json` - долг H1.

Как проверять поиск на GCE: `/tmp/search_accept.py` (не в git),
`export METHODIST_TOKEN=…; python3 /tmp/search_accept.py --base http://127.0.0.1:8000`.

### J - фильтры не врут

Один объект имён в UI / FastAPI Query / SQL. `crm_statuses` режется в SQL
(`_CRM_STATUS_SQL_EXPR`), не после пагинации. Hidden-хосты `#month-*` /
`#yesterday-*` и их рендереры удалены (−370 строк). Фасеты при deep-link:
`ensureFacets()` после любой страницы кроме Обзора (#312). Приёмка: 20 чипов
меняют `total`. PR #311 + #312, релизы 9-10.

### F1 / F2 / F3 - дашборды по одному PR на экран

Паттерн: один GET, один WHERE, один кэш, клик → Найти МО, пустое состояние
с причиной, structure-тест + контрактный pytest + e2e мок. Bugbot по diff
до merge, находки чинятся тестом. Не пушить в ветку после зелёного CI.

F3 Bugbot (починен `047ae5b4` до merge): 1) fallback оставлял старые графики -
`renderDoctor*(null)`; 2) `specialty_median` был AVG - стал `statistics.median`;
3) поле `enough` vs `enough_data` - на fallback `enough = enough_data && !suppressed`.

---

## 8. Как деплоить, не убив backfill

1. Worktree от свежего `origin/main` (основной checkout `/Users/pavelkuzauka/Cursor_Folders/Protocol`
   грязный и отстаёт - не чинить pull/reset).
2. `.env` симлинком: `ln -s /Users/pavelkuzauka/Cursor_Folders/Protocol/.env .env`
   иначе скрипт не видит `GOOGLE_API_KEY`.
3. Мягкий стоп runner, дождаться в логе `stop flag present - exiting`
   **и** `ps … | grep "[b]ash /opt/protocol/deploy/gcp-app/mo_backfill_range"` = 0.
   `pgrep -f mo_backfill_range` ловит сам ssh - ложный «живой».
4. `date -u` не 01:00-04:30. Нагрузка ~0.
5. `bash deploy/gcp-app/deploy_to_gce.sh`. Лог `/tmp/mo_wave/deploy_<sha>.log`,
   фильтр `rg -v LIBARCHIVE`. Скрипт сам откатится, если публичный `/api/version`
   не ответил за 15 с или version/SHA не совпали.
6. Smoke: `/health/live` + `/api/version` = ожидаемые `BUILD_VERSION` и `git_commit`.
7. Приёмка волны на GCE (probe / сверка n / Playwright). Скрипты приёмки живут
   в `/tmp/mo_wave/` и на VM `/tmp/*_accept_*.py` - в git не кладём (токен, PHI).
8. Перезапуск runner:
   `nohup bash /opt/protocol/deploy/gcp-app/mo_backfill_range.sh 2026-01-01 2026-06-30 >> …/gce-mo-backfill.log`.
   Шаг 8 держит ssh - запускать с таймаутом. Resume: дни с `mo_backfill_done_<день>`
   пропускаются.

Playwright: `require(…/Protocol/node_modules/playwright)` CJS, токен из env.
Симлинк `node_modules` из основного checkout в worktree, если его нет.

Откат: предыдущий образ по SHA, тот же набор `-v/-e`, `GIT_COMMIT_SHA` **предыдущего**
релиза (иначе `/api/version` покажет чужой SHA). Ручной откат один (релиз 4),
авто один (попытка 1 релиза 8).

---

## 9. Как работать агенту в этом репо

- Имя ветки: `cursor|codex|hotfix|release/<задача>[-agentN]-pcN`.
- Worktree: `scripts/ops/git_task_start.sh <slug> --pc=pc1 --branch=cursor/<slug>-pc1 --worktree=/Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/<slug>`.
- Перед правкой: `python3 scripts/ops/pr_dashboard.py --files <path>` (exit 1 = занят).
- PR body только `--body-file /tmp/mo_wave/pr_*.md` (heredoc дописывает vendor-атрибуцию).
- Merge: `gh pr merge N --squash --delete-branch` после зелёного CI. Не смотреть
  старый Actions run после нового push.
- `gh pr checks` exit 8 = pending, не ошибка.
- `scripts/normalize_ui_dashes.py` без аргументов (и даже с файлами) переписывает
  **весь** репозиторий. Лишнее возвращать `git checkout --`. В UI/docs только
  короткий дефис с пробелами, не en/em dash.
- Не `ruff` по `mo-app.js` (17 тыс. ложных invalid-syntax).
- `pytest … | tail && git commit` глотает код pytest - проверять `${PIPESTATUS[0]}`.
- BUILD_VERSION: `scripts/ops/bump_build_version.sh short-slug` в том же коммите.
- Не пушить в `main`. Не force-push / amend после push. Не трогать чужой worktree.
- Temp: `/tmp/mo_wave/`. Скриншоты с ФИО врачей в git не класть.

Директива владельца на эту серию: одна волна = один PR = Bugbot = merge = деплой
= приёмка = запись в план §6b/§7 + handoff. Откат - предыдущий образ.

---

## 10. Цифры на дашборде (стиль)

До 27.09 `.kpi-value` был `font-weight: 850` без цвета и наследовал почти чёрный
`--ink`. Теперь токены `--num-moss / lake / clay / heather / rose / slate`
(светлая и тёмная темы), вес 640. Класс `kpi--<tone>` ставит `kpi()` по подписи
(`kpiNature`) и плитки лекарств/анализов по доле. Полоса внимания - по уже
существовавшим `attention-tile--*`. Числа сводки «Найти МО» красятся цветом бара
(оценки - `gradeColor`, специальности - цикл из шести природных).

Файлы: `mo-tokens.css`, `mo-ui.css`, `mo-app.js`. Не трогает F3-логику.

---

## 11. Ошибки, найденные при сборке карты, и план исправлений

Это не «сделать молча в текущем PR». Отдельные PR после деплоя F3+цветов.

### P0 - задеплоит F3 и не сломает CI

F3 уже в `origin/main` (`70adb73d`), CI #318 был зелёный и **не перезапускался**.
Прод всё ещё на F2. Следующий релиз 13: мягкий стоп backfill → `deploy_to_gce.sh`
с `.env` симлинком → приёмка D1-D5 (ранг, heatmap, scatter, профиль, `enough`,
фильтр врача выбирает его, `specialty_median` есть, fallback чистит графики) →
запись в §6b. Не пушить в удалённую ветку #318 (она удалена).

### P1 - долги Bugbot со старых волн (не F3)

Отдельный PR `cursor/mo-redesign-bugbot-debts-pc1`, только `mo_backend.py` + тесты
контракта (тесты - отдельным PR, если гигиена не пустит вместе):

1. `empty_state.pre_total` считался дважды / врал при пустой выборке.
2. `/freshness` не respektирует часть фильтров шапки - свежесть «всего склада».
3. Сортировка `overall_grade` в Python-пути не совпадает с SQL-порядком шкалы.
4. FTS стем даёт ложные (миопия / миопатия) - ужесточить MATCH или вычесть
   стеммы короче 5.
5. Нечёткий поиск пропускается, если уже есть `term_codes` - опечатка после
   выбора чипа кода не ищет.
6. `_search_chip_counts` протекает в `applied_filters` ответа.

### P2 - приёмка и вёрстка

1. Таблица «Найти МО» / Очередь / Врачи вылезает за правый край на 1024
   (15 / 18 колонок). Лечить в F6: sticky первая колонка + явный горизонтальный
   скролл, не резать колонки.
2. Холодный YTD Обзора 8,8 с и сводки 2,56 с. Не откатывали: кэш 120/60 с
   даёт 14 / 9-16 мс. Если WAL backfill сбрасывает кэш слишком часто -
   не считать stamp от каждого append, а от `mtime` после commit дня.
3. Первый удар месяца F2 2,87 с > бюджет 2 с - тот же кэш, не отдельный индекс.

### P3 - данные и шкала (не код UI)

1. Backfill янв-июнь не закончен. День `2026-06-22` повторить.
2. Финальный `recompute` истории пациентов - после всех месяцев (волна C п. финал).
3. `icd_chapter` пуст у 23% сентября - волна I.
4. Две «критично» (плитка P0/P1 vs чип `overall_grade`) и чипы `worst_severity`
   с P-кодами внутри - H2.
5. `critical` в складе июль-сентябрь = 0. Это данные / порог, не баг фильтра.
6. zone1 ok 4,8%, zone2b ok 0%, two-top grades ~95% - H3, только решением владельца.
7. Эмбеддинги «похожие» (D шаг 8) - ночной job на GCE после F.
8. Ревью `dx_aliases_ru.json` врачом - H1.
9. «СД 2»: 3 из 10 строк эвристика скрипта не объясняет (не видит `diagnosis_text`)
   - проверить методисту вручную, не чинить движок вслепую.
10. Rceth: нет недельного re-crawl - решение владельца, не агента.
11. Зомби PR #113 / #186 / #261 закрыть или переоткрыть с `origin/main`
    (дашборд считает их владельцами файлов).

### Порядок починки

1. Деплой релиза 13 (F3, плюс цвета если уже в main).
2. Приёмка Врачи + запись в план.
3. F4 Лекарства (не долги). Долги P1 - параллельным PR, не в F4.
4. F5 → F6 (там overflow) → F7 → G → K.
5. H1/I фоном с недели 2; H2/H3 только после разметки.

---

## 12. Следующая безопасная команда

Проверить, что runner не в середине дня, затем деплой `70adb73d` (или более
новый main, если цвета уже влиты):

```bash
gcloud compute ssh protocol-app --zone=europe-central2-a --command='
date -u; cat /proc/loadavg
ps aux | grep "[b]ash /opt/protocol/deploy/gcp-app/mo_backfill_range" || echo RUNNER_GONE
sudo tail -8 /var/data/medical_exams/logs/gce-mo-backfill.log
'
```

Не деплоить, пока в логе идёт `progress N/M` текущего дня. Мягкий стоп →
выход runner → `deploy_to_gce.sh` из worktree на `origin/main` с `.env` симлинком.

Нельзя параллельно трогать: `clinical_knowledge/mo_backend.py`,
`frontend/web/shared/mo-app.js`, `mis-kz-quality.html`, `rag_server.py`
(кроме своей строки `BUILD_VERSION`), `docs/plans/2026-09-26-mo-analytics-redesign-v2.md`
если другой агент уже держит план.
