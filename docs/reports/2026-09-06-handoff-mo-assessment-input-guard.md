# Handoff: assessment input и revision conflict guard

Дата: 2026-09-06.
Branch: `cursor/mo-assessment-input-guard-agent1-pc1`.
Worktree: `/private/tmp/protocol-task-mo-assessment-input-pc1`.
Base: `7673f427f0b2e2b9c1279896b5fa4bfa9d5cff0c`.

## Требование

Закрыть R01 и Wave A: заполненные canonical Dx/code/plan не должны становиться
автоматическими findings об отсутствии из-за пустого или потерянного evaluator
input. Конфликтный вывод хранится для аудита, но не считается подтверждённым и
не попадает в шаблон врачу.

## Реализация

- `evaluate_kz_deep()` поднимает canonical поля из вложенного `clinical` в flat
  evaluator input до расчёта и сохраняет только безопасную карту присутствия
  полей, без дублирования клинического текста.
- Persisted assessment сравнивает current raw revision, явно объявленный evaluator
  input и отсутствие, заявленное finding.
- Потеря заполненного canonical поля в evaluator input даёт `status=error` и
  `evaluator_input_transport_loss`.
- Finding `A_missing_*`/`B_dx_absent`, противоречащий current raw revision, даёт
  `status=conflict` и `finding_conflicts_with_source_revision`.
- Конфликтный finding остаётся в `fact_mo_finding`, но получает
  `penalty_applied=0`, `needs_human=1`; его source ref не включается в
  подтверждённый assessment snapshot.
- Doctor review brief для `error|stale|conflict|insufficient_data|not_applicable`
  не формируется как доступный подтверждённый итог.
- Пороговые значения, primary-флаги и клинические веса не менялись.

## Проверки

- `python3.11 -m py_compile` для трёх изменённых Python-модулей - passed.
- `pytest tests/test_mo_daily_pipeline.py tests/test_kz_deep_eval.py
  tests/test_mo_case_review_brief.py -q` - 62 passed.
- Synthetic nested canonical input: отсутствие Dx/exam/treatment findings не
  создаётся - passed.
- Synthetic source/finding conflict: assessment `conflict`, finding сохранён с
  `penalty_applied=0`, `needs_human=1`, doctor brief недоступен - passed.
- IDE lint и `git diff --check` - passed.

Постоянные synthetic tests E1/E2 публикуются отдельным level-4 PR.

## Production

Не deployed. После runtime и test-only PR нужен exact-main GCE release и smoke
list/detail/review brief без PHI.

## Следующий шаг

Опубликовать отдельный test-only PR для input transport, source conflict и
list/detail/export parity, затем выпустить Wave A.

## Не менять параллельно

- `clinical_knowledge/mo_daily.py`;
- `clinical_knowledge/kz_deep_eval.py`;
- `clinical_knowledge/mo_case_review_brief.py`;
- `rag_server.py` до merge runtime PR.
