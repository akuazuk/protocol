# Handoff: ICD match pipeline v3 phase 6 (light stem)

Дата: 2026-08-08

## Repo

| | |
|--|--|
| branch | `cursor/mo-icd-match-pipeline-p6-pc1` |
| worktree | `/private/tmp/protocol-task-mo-icd-match-pipeline-p6-pc1` |
| plan | v3 фаза 6 [x] |

## Сделано

- `light_stem` / `MO_ICD_LIGHT_STEM` в `clinical_text_similarity`
- directory `title_match_score` использует общие токены
- тесты «живот/живота»; spike-отчёт

## Не сделано

- Embeddings spike (не нужен)
- Включение `MO_ICD_LIGHT_STEM=1` на GCE по умолчанию (оставить off до явного smoke)

## Следующий трек

`docs/plans/2026-08-08-mo-patient-history-bundle-v2.md` - старт с A1 (`patient_key`).
