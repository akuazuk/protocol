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

## launchd

```bash
python3 scripts/manage_mo_daily_launchd.py install
python3 scripts/manage_mo_daily_launchd.py status
python3 scripts/manage_mo_daily_launchd.py uninstall
```

Устанавливаются основной запуск 07:00, retry 10:00 и hourly catch-up. `launchd` использует
системный timezone macOS, поэтому для точного расписания он должен быть `Europe/Minsk`.
Installer не меняет `pmset`. Wrapper держит `caffeinate` только на время pipeline.

Перед первым production-запуском проверить Telegram env, доступ к exporter dependencies,
системный timezone и отсутствие другого тяжёлого SQL-задания в 07:00.
