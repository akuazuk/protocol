# Локальные выгрузки mis_protocol (КЗ из МИС)

Данные **персональные** — в git не коммитятся (`*.parquet`, `*.csv` в `.gitignore`).

## Пересоздать за месяц

```bash
python3 scripts/export_mis_protocol_month.py --month 2026-07
```

Нужен пароль в `~/CURSOR/sql_epam/.env` (`KRAVIRA_DB_PASSWORD`).

Схема полей: `epam/scheme_mis_protocols.docx`, парсер: `clinical_knowledge/mis_protocol_parse.py`.

## На Render (тестовый диск)

Файлы кладутся в `/var/data/mis_protocol/` (persistent disk), не в git.

```bash
# загрузить месяц на Render
bash scripts/render_mis_protocol_data.sh upload 2026-07

# список
bash scripts/render_mis_protocol_data.sh list

# удалить месяц
bash scripts/render_mis_protocol_data.sh delete 2026-07

# удалить всё тестовое
bash scripts/render_mis_protocol_data.sh delete-all
```

На самом Render (SSH):

```bash
ls -lh /var/data/mis_protocol/
rm -f /var/data/mis_protocol/mis_protocol_2026-07.*
# или
rm -rf /var/data/mis_protocol
```
