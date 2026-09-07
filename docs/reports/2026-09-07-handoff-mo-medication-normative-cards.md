# Handoff: MO medication normative cards

Дата: 2026-09-07

## Состояние

- Branch: `cursor/mo-medication-normative-cards-agent1-pc1`
- Worktree: `/private/tmp/protocol-task-mo-medication-cards-pc1`
- Base: `8c6742032a65b651add1d8a2fe42da3fe26fb7d1`
- BUILD_VERSION: `2026-09-07-065417Z-medication-cards`
- PR: [#239](https://github.com/akuazuk/protocol/pull/239)

## Реализация

- Добавлен shadow contract `mo_medication_normative_cards_v1`.
- Каждая карточка содержит assignment, patient assertion/subject/time,
  instruction source/revision, applicability, uncertainty и protocol check.
- При неподобранном КП `protocol_check=not_evaluated`; карточки не публикуют
  protocol verdict и не добавляют источник `national_protocol`.
- После live protocol-suggest карточки пересобираются для того же case scope.
- Rceth instruction, national protocol и local methodology являются разными
  типами источников.
- Local reg55 pack помечен `normative=false`.
- №127 используется только как вспомогательный источник и не получает
  отдельного denominator.
- Text-only protocol candidate больше не включает B4-B6 deep findings.
- Явный `protocol_check=not_evaluated` отключает protocol-dependent deep findings.
- CASE Review показывает назначение, режим, activity, patient fact, источники
  и причины ограничений.

Все карточки `shadow=true`, `primary=false`. Primary flags, clinical weights,
penalties и promotion не менялись.

## Проверки

- Focused medication/reg55/zone/frontend suite: 49 passed.
- Full Playwright: 15 passed.
- Synthetic acceptance: unmatched protocol, matched protocol source, past
  medication, Rceth revision, local pack non-normative и deep protocol gate.
- Python compile, JS syntax, `git diff --check` и IDE diagnostics - успешно.

## Production baseline

- Production SHA: `8c6742032a65b651add1d8a2fe42da3fe26fb7d1`.
- Production version: `2026-09-07-062331Z-medication-guards`.
- Per-assignment medication synthetic smoke внутри production image успешен.
- Normative cards ещё не merged и не deployed.

## Следующий безопасный шаг

Required CI → merge → exact-main GCE deploy → production API/drawer smoke.
После этого отдельный level-4 acceptance PR для Wave E.

## Не трогать параллельно

- `clinical_knowledge/medication_normative_cards.py`
- `clinical_knowledge/mo_reg55_section.py`
- `clinical_knowledge/kz_deep_eval.py`
- `frontend/web/shared/mo-app.js`
- `frontend/web/shared/mo-ui.css`
- `rag_server.py`
