# Handoff: ICD match pipeline v3 phase 4 (LLM grey zone)

Дата: 2026-08-08

## Repo

| | |
|--|--|
| branch | `cursor/mo-icd-match-pipeline-p4-pc1` |
| plan | `docs/plans/2026-08-08-mo-icd-dx-matching-pipeline-v3.md` фаза 4 [x] |

## Сделано

- `clinical_knowledge/mo_icd_llm_review.py` - контракт agree/partial/no, shadow findings
- `scripts/run_mo_icd_llm_review.py` - batch по дню (CSV), out `llm_icd_review/.../reviews.jsonl`
- Night: `mo_llm_range_runner.sh` + scp в `deploy/gcp-llm/run_on_gce.sh`
- Flags default **off**: `MO_ICD_LLM_REVIEW=0`, `MO_ICD_LLM_CLEAR_WEAK=0`
- Не в overall / не в chip; unit-тесты с mock Gemini

## Как включить на GCE (осознанно)

```bash
# на VM / в env range-runner:
export MO_ICD_LLM_REVIEW=1
export MO_ICD_LLM_REVIEW_LIMIT=50
bash deploy/gcp-llm/run_on_gce.sh 2026-08-04 --foreground
# или smoke dry-run внутри container:
# MO_ICD_LLM_REVIEW=1 python scripts/run_mo_icd_llm_review.py --date 2026-08-04 --dry-run --limit 5 --force-enable
```

## Не сделано

- Фаза 5 soft-fill
- Primary для LLM findings (≥30 размеченных)
- UI отдельной колонки

## Следующая безопасная команда

После merge: `bash deploy/gcp-app/deploy_to_gce.sh` (флаг LLM остаётся 0).
Живой Gemini review - только через `run_on_gce.sh` с `MO_ICD_LLM_REVIEW=1`.
