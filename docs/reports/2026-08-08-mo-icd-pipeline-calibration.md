# Калибровка MO ICD match pipeline (фаза 3)

Дата отчёта: 2026-08-08
Скрипт: `scripts/calibrate_mo_icd_pipeline.py`
PHI: не включён (агрегаты + hashed visit_id + id эталонов).

## Важно: справочник на GCE

Первый прогон на контейнере дал accuracy≈0.14: в image не было
`data/icd_reference/icd10_ru_mkb10su.json` (`ru_valid_codes=0`).
После копирования JSON (и фикса Dockerfile/deploy) эталоны = **1.0**,
ниже - повторный прогон с справочником.

## Пороги (snapshot)

| threshold | value |
|--|--|
| `dir_hit_score_min` | 0.12 |
| `name_ok` | 0.42 |
| `name_review` | 0.28 |
| `suggest_min` | 0.08 |
| `text_fit_ok` | 0.35 |
| `text_fit_review` | 0.25 |

## Эталоны (прокси методиста)

- n = **22**
- accuracy (chip) = **1.0**
- precision `not_in_directory` = **1.0**
- recall `not_in_directory` = **1.0**

### Chip histogram (predicted)

| key | n |
|--|--|
| `ok` | 18 |
| `not_in_directory` | 2 |
| `missing_dx` | 1 |
| `weak_name` | 1 |

### Confusion expected → predicted

| expected \ predicted | missing_dx | not_in_directory | ok | weak_name |
|--|--|--|--|--|
| `missing_dx` | 1 | 0 | 0 | 0 |
| `not_in_directory` | 0 | 2 | 0 | 0 |
| `ok` | 0 | 0 | 18 | 0 |
| `weak_name` | 0 | 0 | 0 | 1 |

### Mismatches (etalon id only)

_нет расхождений_

### Finding codes (etalons)

| key | n |
|--|--|
| `B_icd_dir_no_match` | 2 |
| `B_icd_name_no_match` | 2 |
| `B_dx_absent` | 1 |
| `B_icd_dir_code_unknown` | 1 |
| `B_icd_dir_text_mismatch` | 1 |

## Выборка дня `2026-08-04`

- n = **200**
- needs_llm_review rate = **0.685**
- sample visit hashes: `44da18c9e1d2`, `ae4716ba5404`, `a9f0e7953e34`, `c508cc1295ca`, `ee9ff1dd6148`, `efa11adc6897`, `c00ae49aacdd`, `2356f4e5dd78`

### Chip share

| key | n |
|--|--|
| `missing_dx` | 1 (0.005) |
| `not_in_directory` | 73 (0.365) |
| `ok` | 43 (0.215) |
| `weak_name` | 83 (0.415) |

### Pipeline verdict

| key | n |
|--|--|
| `review` | 101 |
| `fail` | 56 |
| `ok` | 43 |

### Top finding codes

| key | n |
|--|--|
| `B_icd_dir_text_mismatch` | 101 |
| `B_icd_name_weak_match` | 69 |
| `B_icd_name_no_match` | 55 |
| `B_icd_dir_code_unknown` | 23 |
| `B_icd_dir_no_match` | 2 |
| `B_icd_mismatch_mis` | 2 |
| `B_dx_absent` | 1 |

### name_fit bins

| key | n |
|--|--|
| `0.28-0.42` | 69 |
| `<0.28` | 59 |
| `0.42-0.60` | 48 |
| `>=0.60` | 24 |
| `na` | 0 |

### text_rubric_fit bins

| key | n |
|--|--|
| `<0.28` | 132 |
| `>=0.60` | 48 |
| `0.42-0.60` | 12 |
| `0.28-0.42` | 8 |
| `na` | 0 |

## Решение по primary

- `MO_ICD_NAME_IN_PRIMARY` → **1**
- `MO_ICD_DIR_IN_PRIMARY` → **0**
- `MO_ICD_PIPELINE_IN_PRIMARY` → **0**
- пороги менять: **нет (дефолты v3)**

Эталон accuracy=1.0, not_in_directory P/R=1.0/1.0. День: missing_dx share=0.005, not_in_directory share=0.365. Включаем **только** `MO_ICD_NAME_IN_PRIMARY=1` (мягкая ось). `MO_ICD_DIR_IN_PRIMARY` и `MO_ICD_PIPELINE_IN_PRIMARY` оставляем 0 до ручной разметки дня и контроля overall.

## Следующий шаг

- Фаза 4: LLM review только для `needs_llm_review` (GCE, флаг off).
- После ручной разметки ≥20 живых визитов - пересмотреть DIR primary.
