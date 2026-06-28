# Chunk Quality Report (2026-06-28)

## Summary
- Chunks read: **62315**
- Flagged (score < threshold or issues): **29642**
- Avg quality_score: **0.935**
- Median quality_score: **1.0**

## Issue counts

| Issue | Count |
|-------|------:|
| `weak_section_title` | 12796 |
| `truncated_list` | 7576 |
| `too_long` | 5893 |
| `type_body_but_clinical` | 4451 |
| `too_short` | 3682 |
| `empty_entities` | 2531 |
| `preamble_leak` | 645 |

## Chunk types (top)

| Type | Count |
|------|------:|
| `body` | 22180 |
| `treatment` | 15172 |
| `drug_list` | 8004 |
| `diagnostics` | 7638 |
| `criteria_block` | 2619 |
| `terms` | 2010 |
| `pharmacotherapy` | 1331 |
| `classification` | 1322 |
| `dispensary` | 620 |
| `prevention` | 463 |
| `rehabilitation` | 351 |
| `routing` | 328 |
| `algorithm` | 268 |
| `appendix` | 9 |

## Indexable

- `True`: 59975
- `False`: 2340

## Samples

- **low_score**: 00335c08f8a48efeccbecca3_s6_c1, 00335c08f8a48efeccbecca3_s6_c3, 00335c08f8a48efeccbecca3_s11_c3, 00335c08f8a48efeccbecca3_s14_c1, 00335c08f8a48efeccbecca3_s15_c8
- **preamble**: 01c177cfb5344ccc02ad2e83_s1_c0, 01c177cfb5344ccc02ad2e83_s5_c0, 01f7bb87c213de13cb9c36ca_s1_c0, 01f7bb87c213de13cb9c36ca_s3_c0, 02ed61600b44b7b1d3f26204_s1_c0
- **icd_inflation**: -
- **body_clinical**: 00335c08f8a48efeccbecca3_s6_c1, 00335c08f8a48efeccbecca3_s6_c3, 00335c08f8a48efeccbecca3_s10_c3, 00335c08f8a48efeccbecca3_s11_c3, 00335c08f8a48efeccbecca3_s14_c1

## vs baseline

- Baseline avg score: 0.915
- Delta avg: 0.02
