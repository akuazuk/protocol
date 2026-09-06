# Handoff: групповые знаменатели лекарств и анализов

Дата: 2026-09-06. Roadmap: A10/A11, backend contract.

Branch `cursor/mo-family-group-denominators-agent1-pc1`.
Worktree `/private/tmp/protocol-task-mo-family-group-denominators-pc1` (locked).
Base при старте: `a204df66851afe1d93f3ce05a880127aa97efaab`.

## Требование и реализация

Раньше `by_doctor` и `by_specialty` возвращали число проблем группы, но процент
делили на все МО периода. Такой показатель является вкладом группы в общий
поток проблем, а не долей проблем у врача или специальности.

Backend теперь отдельно возвращает:

- `problem_cases` и `group_cases`;
- `problem_pct_of_group`;
- прежний показатель как `period_contribution_pct`;
- явные `denominator_kind` и `denominator_n`;
- `evaluated_cases` и `problem_pct_of_evaluated` как `null`, пока evaluator
  coverage не хранится на уровне каждого случая;
- `small_n`, `comparison_min_n=20`, `ranking_eligible=false` и причину запрета
  сравнения.

Это намеренно не превращает неизвестный знаменатель оценимых случаев в число.
Следующий frontend PR должен показать обе семантики и не ранжировать строки до
доступности evaluator coverage. Clinical weights и scores не меняются.

## Проверки

- 21 focused backend/family test - passed;
- ruff check - passed;
- synthetic SQLite: 1 problem из 2 случаев группы даёт group rate 50%,
  contribution 50%, evaluated unknown, ranking false;
- `git diff --check`, branch guard и IDE lint - passed.

`ruff format --check` всего `mo_backend.py` сообщает существующее
repo-wide форматирование файла; применять механическое форматирование к 5,500+
строк в этом PR нельзя. Diff проходит lint и не форматирует несвязанный код.

На момент первой записи: implemented и locally verified; CI, merge, deploy и
frontend presentation ещё не выполнены. Постоянный backend regression должен
быть отдельным test-only PR уровня 4.
