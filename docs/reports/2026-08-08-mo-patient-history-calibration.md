# Калибровка истории пациента (фаза C)

Дата: 2026-08-08  
План: `docs/plans/2026-08-08-mo-patient-history-bundle-v2.md`

## Флаги (defaults)

| Env | Default | Смысл |
|--|--|--|
| `MO_PATIENT_HISTORY_BUNDLE` | `1` | собирать бандл + shadow finding |
| `MO_PATIENT_HISTORY_IN_PRIMARY` | `0` | не влияет на overall |
| `MO_PATIENT_HISTORY_LOOKBACK_DAYS` | пусто = весь склад | опциональный потолок |

## После deploy на GCE

```bash
python scripts/recompute_mo_days.py \
  --data-root /var/data/medical_exams \
  --first-date 2026-08-04 --last-date 2026-08-04 \
  --warehouse /var/data/medical_exams/warehouse/mo_analytics.sqlite

sqlite3 /var/data/medical_exams/warehouse/mo_analytics.sqlite \
  "SELECT history_tier, COUNT(*) FROM fact_mo_case
   WHERE visit_date='2026-08-04' GROUP BY 1 ORDER BY 2 DESC"

sqlite3 ... "SELECT COUNT(*) FROM fact_mo_case WHERE patient_key IS NOT NULL AND patient_key!=''"
```

Цели калибровки: доля `insufficient` низкая при наличии patient_id в CSV;
бейджи в дне и блок в разборе без сырого patient_id в JSON истории.
