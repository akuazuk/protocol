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
