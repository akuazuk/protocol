# Handoff: МО — ограничения текстовой проверки инструкций ЛС

2026-09-06; akuazuk/protocol; agent1 / pc1.
Branch codex/mo-label-assertion-guards-agent1-pc1.
Worktree /Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/mo-label-assertion-guards (locked).
Base fe0734a8a5956d1e7a8d494da895319411968d01; HEAD — commit с этим handoff, SHA в PR.

## Исправлено

Детский подраздел дозирования больше не задаёт верхний возраст всего препарата.
Явное нижнее возрастное противопоказание сохранено; evidence берётся из
противопоказаний. Heart/renal/hepatic failure требует орган и недостаточность
в одной фразе, не одного упоминания органа. Очевидные отрицательные,
неопределённые и семейные фразы исключаются из поиска противопоказаний.
Границы исходных полей сохранены, чтобы отрицание в анамнезе не поглощало диагноз.
Все результаты текстового label-check — candidate, needs_human=true, shadow;
названия не утверждают доказанное нарушение по простому совпадению слов.
Engine marker v2. Primary flags не менялись.

## Проверка и пределы

28 passed: test_rceth_label_findings, test_rceth_sync, test_mo_family_scores.
Ruff, py_compile, diff --check passed. Полный CI ожидается.
Это ограниченный консервативный sentence guard, не полный assertion/subject/
temporality parser. Неоднозначные фразы целиком пропускаются, поэтому возможны
пропуски; отсутствие findings не означает «противопоказаний нет».
Объект препарата/форма/редакция на дату, дозировка конкретного назначения,
индикационный граф и клиническая валидация остаются. A03 частично, A18 устранён;
A19 только честное представление candidate. Старые результаты не пересчитаны.

## Состояние и синхронизация

BUILD_VERSION 2026-09-06-091126Z-mo-label-assertion-guards.
Merge/deploy нет. Последний проверенный production a592d588, health/version ok;
main на старте уже включает #205 (fe0734a8). Связанный audit/tracker #209.
Держим clinical_knowledge/rceth_label_findings.py, tests/test_rceth_label_findings.py
и этот handoff. В rag_server.py только BUILD_VERSION. Мержить последовательно
после предыдущих МО PR, синхронизировав с main без force-push; повторить тесты.
Не удалять locked worktree и не включать новые primary правила без clinical gate.

Следующая безопасная команда:

```bash
gh pr list --repo akuazuk/protocol --state open
```
