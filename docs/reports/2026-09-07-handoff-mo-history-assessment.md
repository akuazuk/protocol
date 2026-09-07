# Handoff: MO history assessment context

Дата: 2026-09-07

## Состояние

- Branch: `cursor/mo-history-assessment-context-agent1-pc1`
- Worktree: `/private/tmp/protocol-task-mo-history-context-pc1`
- Base: `57c0b1b28cb338767af7ec1f3b33813e1d3435ad`
- BUILD_VERSION: `2026-09-07-051321Z-history-assessment`
- PR: [#236](https://github.com/akuazuk/protocol/pull/236)

## Реализация

- Добавлен text-free shadow contract `mo_history_assessment_v1`.
- Разделены `history_available`, `any_prior_exists`,
  `relevant_episode_prior_exists` и `correction_assessable`.
- Публикуются причины исключения: история недоступна, prior отсутствует,
  prior относится к другому эпизоду, нет сравнимого прошлого плана.
- Longitudinal warehouse и document prior остаются отдельными источниками.
- Generic document prior не включает оценку коррекции без явной episode relevance.
- Rubric МЗ и критерии коррекции №55 получают единый assessability guard.
- CASE Review показывает, почему коррекция плана оценивается или остаётся н/д.
- `cutoff_at` assessment передаётся в history contract; существующая выборка prior
  по-прежнему исключает текущую дату и будущие визиты.

Вес, пороги, primary/shadow и клинические формулы не менялись. При
`correction_assessable=true` прежний алгоритм сравнения плана сохранён.

## Проверки

- Focused history/rubric/zone/reg55/frontend suite: 63 passed.
- Synthetic semantics: no history, unrelated rich prior, relevant episode prior,
  correction guard и unchanged shadow primary - успешно.
- Python compile, JS syntax, `git diff --check` и IDE diagnostics - успешно.
- Ruff локально недоступен; обязательный CI выполнит canonical lint.

## Production baseline

- Production SHA: `57c0b1b28cb338767af7ec1f3b33813e1d3435ad`.
- Production version: `2026-09-07-043518Z-drawer-save-safety`.
- Exact version/commit, review-pack schema, unsaved guard и reports smoke успешны.
- History assessment change ещё не merged и не deployed.

## Следующий безопасный шаг

Required CI → merge → exact-main GCE deploy → production read-only history smoke.
Затем отдельный runtime PR для lab identity/result lifecycle B4-B5.

## Не трогать параллельно

- `clinical_knowledge/mo_history_assessment.py`
- `clinical_knowledge/mo_rubric_mz.py`
- `clinical_knowledge/mo_reg55_section.py`
- `clinical_knowledge/mo_zone_scores.py`
- `frontend/web/shared/mo-app.js`
- `rag_server.py`
