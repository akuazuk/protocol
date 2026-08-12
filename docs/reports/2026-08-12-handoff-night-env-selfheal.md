# Handoff: night 2026-08-11 missing + env self-heal

Дата: 2026-08-12
Причина: cron user `pavel` не читал `/opt/protocol/.env.mis` (owner
`pavelkuzauka` после deploy). Повтор той же поломки, что 10.08.
Дополнительно: после успешного score запись `llm_skip` в root-owned
`secure_cases/` валила `set -e` до `write_status`.

## Сделано

- Выгрузка 2026-08-11 вручную: 584 rows / 452 cases, coverage 100%,
  `gce_night_2026-08-11.json` status=success
- Night self-heal: chown env если нечитаемо; SM defaults без .env.mis;
  `llm_skip` через sudo; check-скрипт не падает на env
- assemble/deploy chown cron user `pavel`

## Next

Merge PR self-heal. Следующая ночь 02:00 UTC должна писать вчера без ручного
chown.
