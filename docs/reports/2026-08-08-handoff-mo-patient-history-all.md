# Handoff: patient history bundle A1–C

Дата: 2026-08-08

## Repo

| | |
|--|--|
| branch | `cursor/mo-patient-history-all-pc1` |
| worktree | `/private/tmp/protocol-task-mo-patient-history-all-pc1` |
| plan | `2026-08-08-mo-patient-history-bundle-v2.md` A1–C |

## Сделано

- `mo_patient_history_bundle.py`: бандл, tiers, одно finding, cache upsert
- warehouse: `patient_key`, `doctor_id`, `diagnosis_text`, `history_*`, индексы, cache table
- deep_eval + case detail live merge; UI бейдж + блок «История пациента»
- B0–B3: name thresholds, concordance line-break, LLM queue boost
- тесты `tests/test_mo_patient_history_bundle.py`

## После deploy

Recompute дня + `GROUP BY history_tier` (см. calibration report).
`MO_PATIENT_HISTORY_IN_PRIMARY` оставить `0` до калибровки.
