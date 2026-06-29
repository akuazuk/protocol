# Chunk Quality Report (2026-06-29)

## Summary
- Chunks read: **57852**
- Flagged (score < threshold or issues): **24988**
- Avg quality_score: **0.942**
- Median quality_score: **1.0**

## Issue counts

| Issue | Count |
|-------|------:|
| `weak_section_title` | 11094 |
| `too_long` | 6586 |
| `truncated_list` | 3970 |
| `too_short` | 3223 |
| `type_body_but_clinical` | 3053 |
| `empty_entities` | 2371 |
| `preamble_leak` | 637 |

## Chunk types (top)

| Type | Count |
|------|------:|
| `body` | 18705 |
| `treatment` | 14410 |
| `diagnostics` | 7806 |
| `drug_list` | 7805 |
| `criteria_block` | 2542 |
| `terms` | 1721 |
| `classification` | 1249 |
| `pharmacotherapy` | 1177 |
| `dispensary` | 570 |
| `prevention` | 397 |
| `rehabilitation` | 336 |
| `routing` | 308 |
| `algorithm` | 262 |
| `treatment_plan` | 55 |
| `diagnosis` | 38 |
| `table` | 28 |
| `diagnostics_and_treatment` | 27 |
| `definitions` | 18 |
| `mixed` | 14 |
| `diagnosis_and_treatment` | 14 |

## Indexable

- `True`: 55749
- `False`: 2103

## Samples

- **low_score**: 00335c08f8a48efeccbecca3_s6_c1, 00335c08f8a48efeccbecca3_s6_c3, 01c177cfb5344ccc02ad2e83_s1_c0, 01c177cfb5344ccc02ad2e83_s5_c0, 01c177cfb5344ccc02ad2e83_s7_c0
- **preamble**: 01c177cfb5344ccc02ad2e83_s1_c0, 01c177cfb5344ccc02ad2e83_s5_c0, 01f7bb87c213de13cb9c36ca_s1_c0, 01f7bb87c213de13cb9c36ca_s3_c0_llm_m0, 02ed61600b44b7b1d3f26204_s1_c0
- **icd_inflation**: -
- **body_clinical**: 00335c08f8a48efeccbecca3_s6_c1, 00335c08f8a48efeccbecca3_s6_c3, 01c177cfb5344ccc02ad2e83_s29_c0, 01c177cfb5344ccc02ad2e83_s32_c2, 01c177cfb5344ccc02ad2e83_s32_c3

## vs baseline

- Baseline avg score: 0.915
- Delta avg: 0.027
