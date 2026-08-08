# Handoff: ICD match pipeline v3 phase 3

Дата: 2026-08-08

## Repo / Git

| | |
|--|--|
| branch | `cursor/mo-icd-match-pipeline-p3-pc1` |
| base | `origin/main` @ phase 1-2 handoff |
| plan | `docs/plans/2026-08-08-mo-icd-dx-matching-pipeline-v3.md` фаза 3 [x] |
| calibration | `docs/reports/2026-08-08-mo-icd-pipeline-calibration.md` |

## Сделано

1. `mo_icd_thresholds.py` - единые пороги + env override + `MO_ICD_PIPELINE_IN_PRIMARY`.
2. `scripts/calibrate_mo_icd_pipeline.py` + `eval/mo_icd_pipeline/etalon_labels_v1.jsonl` (22 эталона).
3. Прогон GCE день `2026-08-04` n=200 (агрегаты, hashed visit_id).
4. Решение primary: **NAME=1**, DIR=0, PIPELINE=0 (в `deploy_to_gce.sh` defaults).
5. **Fix:** RU ICD JSON в Docker image / deploy tar (раньше `ru_valid_codes=0` на GCE).

## Не сделано

- Фаза 4 LLM review
- DIR primary (ждёт ручных labels)
- Soft-fill warehouse

## Следующая команда после merge

```bash
bash deploy/gcp-app/deploy_to_gce.sh
curl -fsS https://protocol.kravira.by/api/version
# в контейнере: python3 -c "import icd_mkb; print(len(icd_mkb._ru_valid_codes()))"  # ~15616
```
