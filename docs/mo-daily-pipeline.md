# Ежедневный МО-pipeline

Локальный pipeline извлекает календарный день MIS, проверяет raw partition, пересобирает
month-to-date, запускает существующий deterministic/deep batch и создаёт защищённый отчёт
и обезличенный public snapshot.

## Безопасная проверка без SQL и VPN

```bash
python3 scripts/run_mo_daily_report.py --date yesterday --dry-run --no-telegram
python3 scripts/export_mo_daily.py --date 2026-07-27 --out-dir /tmp/mo-export --dry-run
```

`--dry-run` не создаёт state/lock, не запускает subprocess, не обращается к БД и не
читает/изменяет состояние VPN.

## Рабочие команды

```bash
python3 scripts/run_mo_daily_report.py --date yesterday
python3 scripts/run_mo_daily_report.py --date 2026-07-27
python3 scripts/run_mo_daily_report.py --catch-up --first-date 2026-01-01
python3 scripts/run_mo_daily_report.py --reconcile-days 3
python3 scripts/run_mo_daily_report.py --this-week --force
python3 scripts/run_mo_daily_report.py --previous-week
```

Pipeline использует `Europe/Minsk`, блокирует параллельный запуск через `fcntl`, хранит
stage/heartbeat/revision в `data/medical_exams/state/pipeline.json`, повторяет временные
ошибки экспорта и восстанавливает исходное состояние VanyaVPN в `finally`.
Необязательный `MO_SQL_EPAM_STATUS_FILE=/path/to/status.json` добавляет read-only health
signal соседней синхронизации `sql_epam`; pipeline не запускает и не изменяет её.

Raw, quarantine, SQLite, secure case detail, отчёты и runtime public snapshots находятся
в `data/medical_exams/` и исключены из git. Public JSON не содержит patient/visit id,
ФИО врача и клинического текста; малые организационные группы подавляются.

## Ручная проверка артефактов

```bash
python3 scripts/validate_mis_export.py \
  --parquet data/medical_exams/raw/2026/07/mo_2026-07-27.parquet \
  --date 2026-07-27 --source-rows 100

python3 scripts/merge_mis_protocol_export.py \
  --daily-dir data/medical_exams/raw/2026/07 \
  --month 2026-07 \
  --out-dir data/medical_exams/secure_cases/2026/07
```

Blocking gate переносит partition в quarantine и не обновляет warehouse/public current.
Нулевой день допустим только при подтверждённом `source_rows=0`.

## Гейты неполного приёма

Ночной экспорт «вчера» иногда отдаёт десятки строк вместо сотен или приходит без join с
`mis_data`. Такие дни блокируются до scoring и уходят в quarantine:

| Код | Когда блокирует | Порог |
|---|---|---|
| `volume_collapse` | строк меньше доли от медианы того же дня недели (нужно >= 3 исторических дня) | 35% |
| `doctor_join_broken` | на дне >= 20 строк, а ФИО врача заполнено меньше порога | 50% |

Праздники и осознанный перезабор малого дня: `MO_VOLUME_COLLAPSE_RATIO=0` отключает гейт
объёма для конкретного запуска (`volume_anomaly` остаётся предупреждением).

## Статус `partial`: день доделывается

Если экспорт принят, но оценка покрыла не все допущенные записи, день получает статус
`partial`, а не `success`:

- `report.json` и public snapshot содержат `partial: true` и блок `completeness`
 (`coverage_pct`, `reasons`); HTML-отчёт показывает плашку «День доделывается»;
- в витрине `fact_mo_daily.quality_status` = `partial`;
- `--catch-up` (retry 10:00 и hourly) возвращает такой день в работу даже если он лежит
 позади уже успешных дат, и продолжает scoring через `--resume`;
- после 4 попыток день перестаёт переспрашиваться, чтобы не крутиться вечно.

Причины `partial`: `scoring_coverage` (оценены не все допущенные записи),
`scoring_errors` (ошибки оценки), `llm_queue_pending` (остались записи в очереди LLM).

## Витрина: один файл на аналитику и кабинет

`data/medical_exams/warehouse/mo_analytics.sqlite` (в проде `/var/data/medical_exams/warehouse/`) -
единственный файл: звёздная схема ежедневного pipeline и операционные таблицы кабинета
методиста лежат вместе. Раньше кабинет писал в отдельный `mo_methodist.sqlite`, где те же
имена таблиц имели другую схему, а факты дублировались write-only копией, которую никто не читал.

- pipeline заполняет `fact_mo_case`, `fact_mo_finding`, `fact_mo_score_axis`,
 `fact_mo_daily`, `fact_mo_doctor_daily` и все `dim_*` за один день;
- кабинет владеет `crm_case_state`, `crm_case_event`, `saved_view`, `export_job`;
 ключ CRM - `case_id` = `visit_id` МИС (разбор ведётся по визиту);
- pipeline создаёт CRM-заготовки только для случаев из очереди разбора и **никогда**
 не перезаписывает статус, поставленный методистом;
- `dim_diagnosis.chapter` заполняется главой МКБ-10 по коду - для группировки диагнозов
 без внешнего справочника.

Перенос старого файла (идемпотентно, источник только читается):

```bash
python3 scripts/migrate_mo_crm_to_warehouse.py --dry-run
python3 scripts/migrate_mo_crm_to_warehouse.py
```

API выполняет тот же перенос автоматически при первом обращении, если старый файл ещё есть.
CRM-таблицы прежней схемы с данными переименовываются в `*_legacy` и остаются в файле.

## Публикация в прод: как данные попадают на Render

Конвейер работает на рабочей машине, а прод читает диск Render. Без публикации в
`/var/data/medical_exams` страница «МО Аналитика» показывает только старые месячные файлы.

```bash
# витрина + отчёты + состояние + оценки последних дней
python3 scripts/publish_mo_to_render.py --methodist-token "$METHODIST_TOKEN"

# заодно заменить месячные файлы оценок (нужно после пересчёта методики)
python3 scripts/publish_mo_to_render.py --legacy-first-month 2026-01 --legacy-last-month 2026-07

# посмотреть команды, ничего не отправляя
python3 scripts/publish_mo_to_render.py --dry-run --no-verify
```

Что важно знать:

- **CRM прода не затирается.** В прод уезжает копия витрины **без** строк
 `crm_case_state`, `crm_case_event`, `saved_view`, `export_job`, а на стороне прода
 факты и справочники доливаются через `INSERT OR REPLACE`. Статусы и события,
 поставленные методистом в проде, остаются на месте;
- `MO_DATA_ROOT=/var/data/medical_exams` задан в `render.yaml`;
- большие файлы уходят потоком `gzip | ssh`: scp и rsync на файлах в десятки мегабайт
 Render обрывает («Connection closed by remote host»), поток с повторами - нет;
- месячные файлы заменяются с резервной копией рядом (`*_cases.pre_deep_YYYYMMDD.jsonl`);
- launchd-обёртка публикует автоматически после каждого прогона
 (`MO_PUBLISH_TO_RENDER=0` отключает, `bash scripts/run_mo_daily_launchd.sh publish` - только публикация).

Проверка после публикации:

```bash
curl -s -H "X-Methodist-Token: $METHODIST_TOKEN" \
  https://protocol-bimy.onrender.com/api/methodist/mo/freshness | python3 -m json.tool
```

Ожидаем `status: fresh`, `lag_days <= 1` и `roots[0].exists: true`.

## Одна шкала балла: deep-eval

В файле оценок лежит несколько чисел, и раньше витрина брала первое непустое. Ежедневный
прогон пишет в верхний `overall_pct` формальный L1-балл (в среднем ~71), а месячные
`--deep-only` прогоны оставляют это поле пустым и кладут оценку в `deep.overall_pct`
(в среднем ~87). Из-за этого тренд рвался на стыке истории и свежих дней: 87 до 26.07 и 71 с 27.07.

Витрина живёт на **deep-шкале**: `deep.overall_pct` (и `deep.status`) - основной балл,
он же согласован с осями `fact_mo_score_axis` и замечаниями `fact_mo_finding`, которые
тоже приходят из `deep`. Верхний `overall_pct` и `core_overall_pct` используются только
как запасной вариант, если deep-блока нет.

Оценка идёт **по визиту**: несколько КЗ одного визита сводятся в один случай
(`n_kz_per_visit`), поэтому документ без своего `mis_id` наследует оценку визита, но
только если на визит есть ровно один случай. Покрытие (`coverage_pct`) считается по
допущенным к оценке записям выгрузки, а не по числу случаев - иначе оценки других типов
документов давали больше 100%.

## Пересчёт истории без выгрузки и LLM

Когда меняется методика агрегации (шкала балла, оси, покрытие), историю приводят к новой
методике из уже сохранённых артефактов - без обращения к МИС и без повторной оценки:

```bash
# дни, у которых есть raw-партиция и оценки в secure_cases
python3 scripts/recompute_mo_days.py --first-date 2026-07-27 --last-date 2026-07-29

# месяцы из общих выгрузок mis_protocol_YYYY-MM.csv (свежие дни не трогаем)
python3 scripts/backfill_mo_warehouse.py --first-month 2026-01 --last-month 2026-07 \
  --through-date 2026-07-26 --no-prune
```

`--no-prune` обязателен, если свежие дни уже ведёт ежедневный конвейер: без него backfill
удалит всё после `--through-date`. `--skip-reports` у `recompute_mo_days.py` обновляет
только витрину, не перезаписывая `report.json` и публичные снапшоты.

## Telegram: один дайджест на прогон

Раньше каждый обработанный день писал отдельное сообщение, и при catch-up за неделю
Telegram превращался в шум. Теперь:

- **одно сообщение на прогон**, проблемы сверху, успешные дни одной строкой;
- если делать нечего (все дни `success`), сообщение **не отправляется**;
- сбойный день не обрывает прогон: остальные даты обрабатываются, а код возврата
 остаётся ненулевым, поэтому retry в 10:00 всё равно сработает.

Пример:

```text
МО, приём 30.07 06:12 Europe/Minsk
Не принято: 27.07 - data-quality gate blocked: ['volume_collapse']
Доделывается: 26.07 - покрытие оценки 62.5% (scoring_coverage)
Готово (2): 25.07, 28.07; строк 1110, оценено 1080
```

## launchd

```bash
python3 scripts/manage_mo_daily_launchd.py install
python3 scripts/manage_mo_daily_launchd.py status
python3 scripts/manage_mo_daily_launchd.py uninstall
```

Устанавливаются:

- основной приём **06:00** Europe/Minsk (вчера + catch-up + reconcile 3 дней; в понедельник ещё `--previous-week`);
- retry **10:00**, если daily status не `success`;
- hourly catch-up;
- понедельничный weekly **11:00** как страховка перезаписи прошлой недели Пн-Вс.

`launchd` использует системный timezone macOS, поэтому для точного расписания он должен быть
`Europe/Minsk`. Installer не меняет `pmset`. Wrapper держит `caffeinate` только на время pipeline.

Приём именно утром (~06:00), а не в начале календарных суток: ночной экспорт «вчера»
часто неполный (мало строк, пустой doctor join). Не запускать параллельно другой тяжёлый
SQL-job в том же окне без необходимости.
