# Handoff: persisted MO assessment run

Дата: 2026-09-06.
Branch: `cursor/mo-assessment-run-agent1-pc1`.
Worktree: `/private/tmp/protocol-task-mo-assessment-run-pc1`.
Base: `3eb10641f6e7a15b669878dab6ec282339cb1aa3`.

## Требование

Этап A1 требует воспроизводимый persisted assessment. До изменения warehouse
хранил отдельные версии scorer/schema и content hash, но не связывал их в один
case-level run/snapshot contract.

## Реализация

- `fact_mo_case` расширен immutable metadata одного автоматического primary run:
  run id, revision, source hash, evaluated/snapshot/cutoff time, methodology и
  evaluator versions.
- Execution status отделен от числового результата:
  `completed|partial|insufficient_data|not_applicable|error`.
- Сохраняются reason codes, безопасные source refs, coverage отдельных проверок,
  protocol id/version/applicability и полный text-free snapshot contract v1.
- `evaluation_run_id` детерминирован по daily run, case, source hash и версиям.
  Повторный idempotent upsert того же запуска не создает другую идентичность.
- Legacy warehouse получает новые колонки через `_ensure_columns`; старые runs не
  переписываются до штатного адресного recompute.

Clinical scores, weights, findings и primary flags не менялись. Snapshot пока
только записывается; read-path list/detail/export подключается следующими PR.

## Проверки

- `python3.11 -m py_compile clinical_knowledge/mo_daily.py` - passed.
- `pytest tests/test_mo_daily_pipeline.py -q` - 48 passed.
- Synthetic warehouse smoke: revision 7, status completed, source hash 64 hex,
  snapshot contract v1 и `protocol_applicability_status=not_evaluated`.
- IDE lint и `git diff --check` - passed.

Постоянный acceptance test для snapshot будет отдельным test-only PR уровня 4.

## Production

Этот PR требует обычный exact-main GCE release после merge. Массовый recompute и
backfill не запускать: новые metadata появятся у следующих штатных daily runs.

## Следующий шаг

Подключить persisted snapshot к единому read adapter для list/detail/export,
затем добавить stale/conflict и canonical input guards.

## Не менять параллельно

- `clinical_knowledge/mo_daily.py`;
- `rag_server.py` до merge этого PR.
