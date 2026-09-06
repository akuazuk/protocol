# Handoff: МО — целостность family scores, этап 1c

2026-09-06; akuazuk/protocol; agent1 / pc1.
Branch codex/mo-family-score-integrity-agent1-pc1.
Worktree /private/tmp/protocol-task-mo-family-score-integrity-pc1.
Base a592d588fdd7eb428161024ad13e4e3948bb3754; HEAD — commit с этим handoff, SHA в PR.

## Изменение

Пустой набор findings больше не доказывает score=100: score=null,
status=not_evaluated. Есть явный completed_families contract для будущих
проверенных evaluators; сейчас callers не объявляют полноту. При findings
статус partial. Дубликат primary/shadow по fingerprint или предметному
содержимому штрафуется один раз; разные target остаются разными.
Shadow collection сохраняет режим даже без item flag. Missing primary score
не заменяется shadow score при blend в overall. Lab fallback возвращает
фактический denominator=total_cases вместо неверного cases_with_lab.

## Проверки

31 passed: test_mo_family_scores, test_mo_finding_families, test_kz_deep_eval,
test_mo_meds_labs_dashboards. Synthetic empty/completed, promotion, separate
targets, shadow leakage, denominator. Ruff/diff --check пройдены.
Полный CI ожидается; глобальные baseline failures не объявлены отсутствующими.

## Состояние

BUILD_VERSION 2026-09-06-085436Z-mo-family-score-integrity.
Первый commit: merge/deploy не выполнялись, primary flags и данные не изменялись.
Последний independently verified production: a592d588, health/version ok.
Не считать этап клинической валидацией. Старые persisted scores автоматически
не пересчитываются. UI пока может скрыть обе null-подоси — явный текст
«недостаточно данных» требуется отдельным frontend PR.
Остаются zero/readiness, group denominators, provenance в SQL, clinical gold,
полная identity модель findings и защита от противоречивых duplicate versions.

## Координация

Держим clinical_knowledge/mo_finding_families.py, tests/test_mo_family_scores.py
и этот handoff. С #205/#206 пересечение только нашей BUILD_VERSION строкой.
Мержить после #205/#206, пересобрать на свежем main и прогнать тесты.
Не переписывать API #205 или lab #206. Новые правила и UI отдельными PR.

Следующая безопасная команда:

```bash
gh pr list --repo akuazuk/protocol --state open
```

## Синхронизация после #206

Main e15ac9cf включён merge-коммитом без force-push. API #205 и lab #206
сохранены; единственный конфликт версии разрешён штатным helper.
BUILD_VERSION 2026-09-06-092138Z-mo-family-score-integrity.
Повторная проверка включает family, API cohort и clinical lab bundle;
общий CI должен пройти на новом HEAD. Production ещё не менялся нами.
