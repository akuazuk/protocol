# Handoff: MO medication assignment guardrails

Дата: 2026-09-07

## Состояние

- Branch: `cursor/mo-medication-guardrails-agent1-pc1`
- Worktree: `/private/tmp/protocol-task-mo-medication-guards-pc1`
- Base: `cc460421a8dc109f32f5c674467027357facd1ba`
- BUILD_VERSION: `2026-09-07-062331Z-medication-guards`
- PR: [#238](https://github.com/akuazuk/protocol/pull/238)

## Реализация

- Medication parser публикует per-assignment form, dose, route, duration,
  activity, assertion, subject и fact time.
- Safety evaluation использует только active + confirmed + patient assignments.
- Past medication, negation, family fact и hypothesis не считаются активным
  назначением пациента.
- OR-альтернативы удаляются до DDI и duplicate extraction.
- Dose binding выполняется отдельно для каждого INN: доза препарата A больше
  не закрывает missing-dose finding препарата B.
- DDI, high-alert, NSAID и class duplicate используют один active treatment set.
- Существующая topical DDI demotion сохранена.

Primary/shadow flags, penalties, thresholds и clinical promotion не менялись.
Изменение только предотвращает ложное применение существующих правил.

## Проверки

- Medication/parser/safety/rceth/deep suite: 52 passed.
- Synthetic acceptance: dose bleed, OR alternative, past medication, negation,
  family subject и hypothesis - успешно.
- Python compile, `git diff --check` и IDE diagnostics - успешно.

## Production baseline

- Production SHA: `cc460421a8dc109f32f5c674467027357facd1ba`.
- Production version: `2026-09-07-053910Z-lab-lifecycle`.
- Lab lifecycle schema, CASE contract, image assets и reports smoke успешны.
- Medication guardrails ещё не merged и не deployed.

## Следующий безопасный шаг

Required CI → merge → exact-main GCE deploy → synthetic image check.
Затем отдельный runtime PR: normative card contract, protocol_check gate и
явное разделение national sources от local methodology.

## Не трогать параллельно

- `clinical_knowledge/consult_schema.py`
- `clinical_knowledge/medication_parser.py`
- `clinical_knowledge/medication_safety.py`
- `clinical_knowledge/medication_findings.py`
- `clinical_knowledge/kz_deep_eval.py`
- `rag_server.py`
