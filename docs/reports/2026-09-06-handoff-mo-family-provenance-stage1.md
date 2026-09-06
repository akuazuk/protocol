# Handoff: A23, provenance семейства findings, этап 1

Дата: 2026-09-06.

Branch `cursor/mo-family-provenance-agent1-pc1`.
Worktree `/private/tmp/protocol-task-mo-family-provenance-pc1` (locked).
Base при старте: `c9d5d502bd7f0468273a72769fd77d1e18adad09`.

## Требование

Family KPI смешивал активные, закрытые, shadow и non-shadow findings. Наличие
non-shadow строки нельзя трактовать как подтверждение методиста. Статус
выполнения evaluator в warehouse сейчас не сохраняется.

## Реализация этапа 1

- В KPI входят только активные строки `passed=0`.
- Для каждого семейства добавлен `finding_provenance` с числом уникальных
  случаев: active, non-shadow, shadow, penalty applied и needs human.
- Возвращаются сохранённые `source_ref` и `trust_level`; список источников
  ограничен 50 значениями и имеет явный признак усечения.
- Review counts имеют `null` и `status=not_projected`. Система не выдаёт
  отсутствие CRM-решения за подтверждение или отклонение.
- Старые схемы без `is_shadow`, недоступный warehouse и ошибка запроса
  отражаются отдельными status.

Clinical scores, weights, flags и данные не меняются. `non_shadow_cases` -
технический путь вычисления, не врачебное подтверждение.

## Проверки

- Focused family pytest: 11 passed.
- Ruff check и `git diff --check` - passed.
- Synthetic SQLite acceptance: active=2, non-shadow=1, shadow=1,
  penalty=1, needs-human=1; строка `passed=1` исключена; review остаётся null.

Постоянный regression для synthetic provenance должен идти отдельным test-only
PR уровня 4. Следующий A23 этап после этого PR: read-only CRM join и buckets
candidate/confirmed/rejected/needs-more-data с явным unknown.
