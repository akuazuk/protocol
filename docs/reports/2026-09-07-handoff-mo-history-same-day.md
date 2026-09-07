# Handoff: P1-1 история same-day и query_failed

Дата: 2026-09-07.
Repo: `akuazuk/protocol`.
Владелец: agent1 / pc1.
Branch: `cursor/mo-history-same-day-agent1-pc1`.
Worktree: `/private/tmp/protocol-task-mo-history-same-day-pc1`.
Base: `a6955ef2beac5decc48b338f3c0ae79d3479b232`.
`BUILD_VERSION`: `2026-09-07-102246Z-history-same-day`.
PR: открывается этой задачей. Merge и production нет.

Параллельно открыт P0 PR #243 (save lineage). `mo_backend.py` в этом PR не
менялся, чтобы не держать один файл двумя ветками.

## Синтетическое воспроизведение

До правки `visit_date < day` выбрасывал утренний визит того же дня. Тест
`tests/test_mo_history_same_day.py` это фиксирует:

- `v-morning` 09:10 при cutoff 14:00 должен входить;
- `v-evening` 16:40 и `v-unknown` без времени не входят в scored shelves;
- `query_failed` не считается доступной историей.

На текущем main до этого PR первые два теста не собирались (`cutoff_at` не
было), третий падал: `history_available is True`.

## Что сделано

- SQL отбора: `visit_date <= day`, затем классификация по `visit_at`/`cutoff_at`.
- Время старых date-only строк не выдумывается: prior day входит, same-day
  без доказанного timestamp получает `unknown_time` и не даёт correction.
- `excluded_visits` с reason для timeline.
- `ok`/`status`: empty / has_priors / unavailable / error.
- `query_failed` при SQL ошибке; assessment не называет это available.
- Night path в `mo_daily.py` передаёт `visit_at`/`cutoff_at` из raw, если есть.
- Склад без колонки `visit_at` остаётся безопасным: same-day unknown.

Не добавлялась массовая колонка `visit_at` в `fact_mo_case`. Нет выдуманного
времени. Нет правок `mo_backend.py` (ждёт merge #243).

## Проверки

```text
pytest tests/test_mo_history_same_day.py \
       tests/test_mo_patient_history_bundle.py \
       tests/test_mo_history_deep.py \
       tests/test_mo_history_continuity.py \
       tests/test_mo_wave_e_acceptance.py::test_e06_e07_e08_history_assessability_semantics
```

27 passed. `git diff --check` чистый.

## Матрица (дельта этого этапа)

| ID | Было | Стало | Остаток |
|---|---|---|---|
| A20 | open, `visit_date < day` | this_pr: same-day по timestamp | episode graph, timezone канон склада |
| R06 / E07 / E08 | partial | earlier same-day + unknown без credit | document_prior / longitudinal один cutoff в detail API |
| E06 | query_failed = available | this_pr: unavailable/error | - |

Живая полная матрица: `docs/reports/2026-09-07-mo-remaining-acceptance-matrix.md` в #243.
После merge #243 обновить статусы A20/E06 там, не копировать файл во второй PR.

## Следующая безопасная команда

После merge #243 и этого PR, новый worktree от свежего main:

```text
scripts/ops/git_task_start.sh mo-evaluated-denominators --pc=pc1 \
  --branch=cursor/mo-evaluated-denominators-agent1-pc1
```

Туда: evaluated N и projection решения методиста. Не трогать
`mo_review_pack.py`, пока #243 не в main. Не деплоить до exact `origin/main`.
