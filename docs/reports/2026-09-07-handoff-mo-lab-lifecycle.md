# Handoff: MO lab identity and result lifecycle

Дата: 2026-09-07

## Состояние

- Branch: `cursor/mo-lab-result-lifecycle-agent1-pc1`
- Worktree: `/private/tmp/protocol-task-mo-lab-lifecycle-pc1`
- Base: `dfc804692b29cbe61e01b331b5b8dc5ba209cf83`
- BUILD_VERSION: `2026-09-07-053910Z-lab-lifecycle`
- PR: [#237](https://github.com/akuazuk/protocol/pull/237)

## Реализация

- Добавлен shadow contract `mo_lab_result_assessment_v1`.
- Canonical identity разделяет test/order, panel, analyte, specimen и method.
- Отдельно публикуются result status, `available_at`, presence, mention,
  interpretation, action, reference availability и abnormal applicability.
- Same-day результат без `available_at` получает `unknown_same_day`.
- Явный `available_at` после cutoff и результат после даты визита получают
  `post_cutoff`; non-final state не считается готовым результатом.
- Нормальный результат без повторения в диагнозе/плане не создаёт defect.
- Findings допускаются только для abnormal + approved reference + unit +
  provable availability; все результаты остаются shadow и primary weights не меняют.
- Lab warehouse schema расширяется nullable полями `order_ref`, `specimen`, `method`,
  `result_status`, `available_at`. Старый schema читается обратно совместимо.
- UI показывает n доступных, unknown, post-cutoff и actionable результатов.

## Rollout

- Enforcement включается после наличия lifecycle-колонок в lab warehouse.
- Ingest автоматически выполняет additive migration. После production deploy
  release coordinator должен один раз вызвать `_ensure_lifecycle_columns` внутри
  контейнера для текущего mounted `mo_lab.sqlite`, затем выполнить read-only smoke.
- Исторические строки получают `result_status=reported_legacy` при следующем ingest;
  `available_at` остаётся NULL, поэтому same-day остаётся unknown.

## Проверки

- Focused lab/ingest/frontend suite: 60 passed.
- Full Playwright: 15 passed.
- Synthetic acceptance: БАК/culture identity distinct; same-day unknown;
  normal result no defect; abnormal available result actionable; non-final guard.
- Python compile, JS syntax, `git diff --check` и IDE diagnostics - успешно.

## Production baseline

- Production SHA: `dfc804692b29cbe61e01b331b5b8dc5ba209cf83`.
- Production version: `2026-09-07-051321Z-history-assessment`.
- History contract и UI live smoke успешны.
- Lab lifecycle change ещё не merged и не deployed.

## Следующий безопасный шаг

Required CI → merge → exact-main GCE deploy → additive lab schema migration →
synthetic image check и read-only case-detail smoke. Затем Wave C medications.

## Не трогать параллельно

- `clinical_knowledge/mo_lab_result_assessment.py`
- `clinical_knowledge/mo_lab_bundle.py`
- `clinical_knowledge/mo_lab_shadow.py`
- `clinical_knowledge/lab_abnormal_findings.py`
- `clinical_knowledge/lab_unused_findings.py`
- `scripts/ingest_mo_lab_from_mis_tests.py`
- `frontend/web/shared/mo-app.js`
- `rag_server.py`
