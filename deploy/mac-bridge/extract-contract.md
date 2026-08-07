# Контракт: Mac/GCP/BY mis_bridge → app inbound (extract)

Эпохи: E1 upload с Mac, E2 с GCP, E3 с BY. Формат файлов один.

## Input (к МИС)

- Env: `KRAVIRA_DB_PASSWORD`, optional DSN overrides.
- VPN: для текущего host Kravira - VanyaVPN **off** (см. mis-mariadb rule).

## Output (в `$MO_DATA_ROOT/inbound/extract/`)

| Файл | Смысл |
|--|--|
| `mo_YYYY-MM-DD.csv` | дневной срез КЗ/МО (named columns) |
| `mo_YYYY-MM-DD.meta.json` | мета extract |

### meta.json schema (минимум)

```json
{
  "schema_version": 1,
  "day": "2026-08-06",
  "extracted_at": "2026-08-07T12:00:00+00:00",
  "run_host": "mac",
  "row_count": 310,
  "checksum_sha256": "<hex of csv>",
  "source": "kravira_mc.mis_protocol"
}
```

## Запрещено в output

- пароли, `.env`
- полный dump `mis_protocol` без окна дат
- CRM / warehouse sqlite

## Потребитель

`mo_pipeline` на GCP (E1) читает `inbound/extract/`, строит `secure_cases`, ставит LLM job.
