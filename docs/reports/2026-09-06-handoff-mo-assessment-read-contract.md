# Handoff: единый read contract assessment

Дата: 2026-09-06.
Branch: `cursor/mo-assessment-read-contract-agent1-pc1`.
Worktree: `/private/tmp/protocol-task-mo-assessment-read-pc1`.
Base: `fd89e6a0b3a6d0895679c6d09c882a1ea30e8411`.

## Требование

Этапы A3-A4 требуют, чтобы list, detail и export читали один persisted assessment
run, а stale/conflict не выглядели подтвержденным результатом.

## Реализация

- Добавлен единый `_assessment_contract_from_row()` для warehouse и JSONL legacy.
- List, detail record, detail root и cases export возвращают один и тот же
  assessment contract и `evaluation_run_id`.
- Контракт содержит revision/source lineage, methodology/evaluator versions,
  execution status, value, `confirmed_value`, coverage, reason/evidence refs,
  protocol applicability и все unrounded snapshot scores.
- Legacy rows без run metadata явно получают `status=stale`,
  `legacy_projection=true`, `confirmed_value=null`.
- Несовпадение source/content hash дает `stale`; несовпадение persisted snapshot
  и SQL projection дает `conflict`. В обоих случаях `usable_as_confirmed=false`
  и `confirmed_value=null`, исходное value остается доступно для аудита.
- Clinical score, stored finding и CRM status не переписываются.

## Проверки

- `python3.11 -m py_compile clinical_knowledge/mo_backend.py` - passed.
- `pytest tests/test_mo_backend.py tests/test_mo_cohort_contract.py
  tests/test_mo_phase78_reports.py -q` - 41 passed.
- Synthetic list/detail/export smoke: contracts полностью равны для одного run.
- Synthetic mutation SQL projection: status становится `conflict`, confirmed value
  отсутствует.
- IDE lint и `git diff --check` - passed.

Постоянный parity/conflict test публикуется отдельным level-4 PR.

## Production

После merge нужен exact-main GCE release. Старые production rows будут помечены
legacy/stale до штатного recompute; сохраненный score остается видимым как value,
но не как confirmed value. Массовый backfill в этом PR не выполняется.

## Следующий шаг

Добавить canonical input guard: заполненный документ и пустой evaluator input
должны давать `error|insufficient_data`, а не finding об отсутствии поля.

## Не менять параллельно

- `clinical_knowledge/mo_backend.py`;
- `rag_server.py` до merge этого PR.
