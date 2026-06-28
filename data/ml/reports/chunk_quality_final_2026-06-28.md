# Chunk Quality Report (2026-06-28)

## Summary
- Chunks read: **59045**
- Flagged (score < threshold or issues): **26772**
- Avg quality_score: **0.936**
- Median quality_score: **1.0**

## Issue counts

| Issue | Count |
|-------|------:|
| `weak_section_title` | 11811 |
| `too_long` | 6427 |
| `truncated_list` | 4908 |
| `type_body_but_clinical` | 4043 |
| `too_short` | 3706 |
| `empty_entities` | 2423 |
| `preamble_leak` | 642 |

## Chunk types (top)

| Type | Count |
|------|------:|
| `body` | 20402 |
| `treatment` | 14586 |
| `drug_list` | 7895 |
| `diagnostics` | 7407 |
| `criteria_block` | 2574 |
| `terms` | 1796 |
| `classification` | 1246 |
| `pharmacotherapy` | 1236 |
| `dispensary` | 585 |
| `prevention` | 405 |
| `rehabilitation` | 336 |
| `routing` | 303 |
| `algorithm` | 254 |
| `appendix` | 9 |
| `chapter_title` | 4 |
| `section_title` | 3 |
| `drug_description` | 2 |
| `table` | 2 |

## Indexable

- `True`: 56738
- `False`: 2307

## Samples

- **low_score**: 00335c08f8a48efeccbecca3_s6_c1, 00335c08f8a48efeccbecca3_s6_c3, 00335c08f8a48efeccbecca3_s11_c3, 00335c08f8a48efeccbecca3_s14_c1, 00335c08f8a48efeccbecca3_s15_c8
- **preamble**: 01c177cfb5344ccc02ad2e83_s1_c0, 01c177cfb5344ccc02ad2e83_s5_c0, 01f7bb87c213de13cb9c36ca_s1_c0, 01f7bb87c213de13cb9c36ca_s3_c0_llm_m0, 02ed61600b44b7b1d3f26204_s1_c0
- **icd_inflation**: -
- **body_clinical**: 00335c08f8a48efeccbecca3_s6_c1, 00335c08f8a48efeccbecca3_s6_c3, 00335c08f8a48efeccbecca3_s10_c3, 00335c08f8a48efeccbecca3_s11_c3, 00335c08f8a48efeccbecca3_s14_c1

## vs baseline

- Baseline avg score: 0.915
- Delta avg: 0.021
