# Handoff: МО — выбор предыдущего визита эпизода

2026-09-06; akuazuk/protocol; agent1 / pc1.
Branch codex/mo-history-prior-selection-agent1-pc1.
Worktree /private/tmp/protocol-task-mo-history-prior-selection-pc1.
Base a592d588fdd7eb428161024ad13e4e3948bb3754; HEAD — commit с этим handoff, опубликованный SHA в PR.

## Изменение

pick_episode_prior повторно применяет тот же episode matcher, которым определена
continuity, перед загрузкой clinical slots. Несвязанный более полный визит
не подменяет релевантный. Сортировка по дате до limit; выбранный prior — самый
свежий из загруженных релевантных, полнота используется при одинаковой дате.
Если эпизод не найден, unrelated slots не загружаются. Selection policy в payload.

Сохраняется merge Cursor #148: billed key runner не менялся. Official overall
и primary flags не менялись. Это исправление A21, не полноценная longitudinal
модель: ICD-stem matcher всё ещё эвристический; разделение эпизодов во времени,
доступность внутри дня и активная терапия требуют следующих PR и валидации.

## Проверки

23 passed: test_mo_history_deep, test_mo_history_continuity,
test_mo_patient_history_bundle. Новые synthetic regressions: unrelated rich,
recency before completeness, limit после отбора, отсутствие matched episode.
Ruff/diff --check пройдены; полный CI ожидается.
BUILD_VERSION 2026-09-06-085656Z-mo-history-prior-selection.
Первый commit: merge/deploy нет; schema/data migrations нет, backfill/LLM не запускались.
Последний проверенный production a592d588, health/version ok.

## Координация

Не менять параллельно clinical_knowledge/mo_history_deep.py,
tests/test_mo_history_deep.py и этот handoff. Файлы свободны по PR dashboard.
Порядок после #205/#206/#207. Общий rag_server.py меняет только BUILD_VERSION.
После чужого merge синхронизировать свою ветку без force-push, rerun tests,
проверить актуальный CI SHA. Не объявлять весь план выполненным: впереди
№55/evidence, расширенные фильтры, лекарственные guards, UI, clinical gold.

Следующая безопасная команда:

```bash
gh pr list --repo akuazuk/protocol --state open
```
