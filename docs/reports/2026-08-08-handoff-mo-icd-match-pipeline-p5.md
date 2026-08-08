# Handoff: ICD match pipeline v3 phase 5 (warehouse soft-fill)

Дата: 2026-08-08

## Repo

| | |
|--|--|
| branch | `cursor/mo-icd-match-pipeline-p5-pc1` |
| worktree | `/private/tmp/protocol-task-mo-icd-match-pipeline-p5-pc1` |
| base | `origin/main` @ `5bedc164` |
| BUILD_VERSION | `2026-08-08-085440Z-icd-softfill-p5` |
| plan | v3 фаза 5 [x]; full-doc P3 [x] |

## Сделано

- `soft_fill_mkb_for_warehouse`: слот → иначе full-doc
- `fact_mo_case.mkb_code_main_source` / `mkb_code_main_slot`
- `upsert_warehouse` soft-fill для KPI/UI; agreement не трогает
- `mo_backend` отдаёт source в API
- тесты: soft-fill unit + warehouse soft-fill без касания agreement

## Recompute smoke (после deploy)

```bash
# на GCE container:
python scripts/recompute_mo_days.py \
  --data-root /var/data/medical_exams \
  --first-date 2026-08-04 --last-date 2026-08-04 \
  --warehouse /var/data/medical_exams/warehouse/mo_analytics.sqlite
# проверка:
sqlite3 ... "SELECT mkb_code_main_source, COUNT(*) FROM fact_mo_case
  WHERE visit_date='2026-08-04' GROUP BY 1"
```

## Не сделано

- Фаза 6 морфология (только если нужно)
- Soft-fill в CSV export (намеренно нет)
