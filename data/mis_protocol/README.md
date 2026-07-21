# Локальные выгрузки mis_protocol (КЗ из МИС)

Данные **персональные** — в git не коммитятся (`*.parquet`, `*.csv` в `.gitignore`).

## Пересоздать за месяц

```bash
python3 scripts/export_mis_protocol_month.py --month 2026-07
```

Нужен пароль в `~/CURSOR/sql_epam/.env` (`KRAVIRA_DB_PASSWORD`).

Схема полей: `epam/scheme_mis_protocols.docx`, парсер: `clinical_knowledge/mis_protocol_parse.py`.

## Файлы

| Файл | Содержимое |
|------|------------|
| `mis_protocol_YYYY-MM.parquet` | все столбцы + `result_raw` |
| `mis_protocol_YYYY-MM.csv` | то же без сырого `result` (для просмотра) |
| `mis_protocol_YYYY-MM.meta.json` | число строк, колонки, период (без ПДн) |
