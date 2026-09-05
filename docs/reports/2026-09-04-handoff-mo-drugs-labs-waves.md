# Handoff: МО drugs/labs scoring waves 1-4

Дата: 2026-09-04

| | |
|--|--|
| Branch | `cursor/mo-drugs-labs-scoring-agent1-pc1` |
| Worktree | `/private/tmp/protocol-task-mo-drugs-labs-scoring-pc1` |
| Plan | `docs/plans/2026-09-04-mo-drugs-labs-scoring-v1.md` |
| BUILD_VERSION | `2026-09-04-165742Z-mo-drugs-labs-waves` |

## Сделано в коде

### Волна 1
- `data/lab_canons/lab_test_canons.json` (≥15 панелей)
- `lab_canons.py`, `lab_unused_findings.py`
- `B_lab_unused_in_dx` / `B_lab_unused_in_plan` (shadow; `MO_LAB_UNUSED_PRIMARY` default 0)
- wired in `mo_lab_shadow.evaluate_lab_for_case`
- gold template `data/mo_gold/drugs_labs_gold_template.json`

### Волна 2
- `data/drug_safety/therapeutic_classes.json` (6 классов)
- class-dup в `_axis_safety` → shadow unless `MO_CLASS_DUP_PRIMARY=1`
- Rceth primary **не** включали (ждёт gold FP)

### Волна 3
- `lab_abnormal_findings.py` + `lab_reference_ranges.json`
- `service_exam_map.py` + catalog seed
- `formulary_findings.py` + seed
- `drug_disease_findings.py` seed
- `B_lab_ordered_not_used`

### Волна 4
- `mo_dual_score.py`, `mo_anomaly_kpis.py`, `mo_risk_adjust.py`
- API `GET /api/methodist/mo/drugs-labs-kpis`
- dual_scores / article_anomalies в `evaluate_kz_deep` result

## Тесты

```text
pytest tests/test_lab_unused_findings.py \
  tests/test_therapeutic_class_dups.py \
  tests/test_lab_abnormal_and_formulary.py \
  tests/test_mo_drugs_labs_wave4.py \
  tests/test_mo_lab_shadow.py \
  tests/test_rceth_label_findings.py \
  tests/test_medication_safety.py \
  tests/test_nsaid_alternatives_topical.py \
  tests/test_medication_findings.py -q
# all passed
```

## Не сделано (нужен человек / ops)

- Разметка gold ≥50/100 методистом и включение primary-флагов
- Rescore/backfill дней topical DDI
- Полный Rceth crawl/parse
- UI-полоски на Обзоре (есть API; фронт - отдельно)
- Deploy

## Primary flags (все default off)

`MO_LAB_UNUSED_PRIMARY`, `MO_CLASS_DUP_PRIMARY`, `MO_RCETH_LABEL_PRIMARY`,
`MO_LAB_ABNORMAL_PRIMARY`, `MO_FORMULARY_PRIMARY`
