# Handoff: МО — неизвестные оценки допуска

2026-09-06; akuazuk/protocol; agent1 / pc1.
Branch: codex/mo-dual-score-unknown-agent1-pc1.
Worktree: /Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/mo-dual-score-unknown (locked).
Base: a592d588fdd7eb428161024ad13e4e3948bb3754. HEAD: commit с этим handoff; точный SHA в PR.

## Изменение и проверка

При отсутствии одной из двух оценок итог допуска теперь null со статусом incomplete;
матрица формы/содержания показывает unknown, а не ошибочно низкое качество.
Нулевой readiness.pct сохраняется и не заменяется альтернативным score.
23 passed: test_mo_dual_score_unknown, test_mo_drugs_labs_wave4, test_kz_deep_eval.
Ruff и git diff --check пройдены. Полный CI ожидается.

## Состояние

BUILD_VERSION: 2026-09-06-090245Z-mo-dual-score-unknown.
Merge/deploy не выполнялись. Последний проверенный production: a592d588,
health/version ok. Persisted оценки не пересчитывались; UI неизвестного
состояния и клиническая валидация остаются отдельными задачами.

Первый временный worktree исчез во время pytest; причина не установлена.
После восстановления в постоянном каталоге все 23 теста прошли.
Активные worktree защищены git worktree lock; не удалять их при cleanup.
Координация размещена в комментарии к #210.

## Синхронизация

Файлы под владением: clinical_knowledge/mo_dual_score.py,
tests/test_mo_dual_score_unknown.py, этот handoff. В rag_server.py только BUILD_VERSION.
Слияние после #205–208 с обновлением от main и повторными проверками.
Не переписывать API, history, lab и family изменения этих PR.
Полный исходный аудит и tracker опубликованы в #209.

Следующая безопасная команда:

```bash
gh pr list --repo akuazuk/protocol --state open
```
