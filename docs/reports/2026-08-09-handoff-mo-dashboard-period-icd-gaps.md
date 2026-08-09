# Handoff: Period ICD + clinical gaps (D5)

Дата: 2026-08-09
Branch: `cursor/mo-dashboard-period-icd-gaps-pc1`
Worktree: `/private/tmp/protocol-task-mo-dashboard-period-icd-gaps-pc1`
Plan: `docs/plans/2026-08-09-mo-dashboards-zones-first-v2.md`

## Сделано

- `/overview`: `icd_visit_status` counts, `clinical_gaps` top codes, `kp_unmatched`.
- Filter `icd_visit_status=` in `_filter_records`.
- Period «Подробнее»: виджеты `#month-icd-status` / `#month-clinical-gaps` + drills.
- D4 checkbox marked done (#87 already in details).

## Координация

| Работа | Действие |
|--|--|
| Draft #77 docs layers | не трогали blueprint/layers files; README не меняли |
| Auth / №55 scorer / drawer | не трогали |

## Тесты

`pytest tests/test_mo_backend.py::test_overview_icd_and_clinical_gaps_secondary tests/test_mo_backend.py::test_period_details_hosts_icd_and_gaps`

## После merge

```bash
bash deploy/gcp-app/deploy_to_gce.sh
```

Counts пустые, если clinical gaps ещё не в `fact_mo_finding` за день - нужен recompute дней после quality-parity.
