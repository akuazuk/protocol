# МО: реализация аудита и синхронизация владельцев

2026-09-06. Канонический scope: [аудит и roadmap](2026-09-06-mo-comprehensive-audit-and-roadmap.md).
Аудит фиксирует bf42e40 и состояние на момент его выполнения. Новые изменения
проверять по этой таблице и GitHub, а не трактовать исторический audit как current production.

## Перед реализацией подтверждено

Cursor merge #148 / 41dd6bb9 (history billed key), #158 / 59c030cb (MIS только GCE),
#204 / a592d588 (requirements-rag.lock deploy allowlist), после #192/#193.
Production version/SHA совпали с a592d588fdd7eb428161024ad13e4e3948bb3754,
version 2026-09-06-073651Z-deploy-lock-allowlist. Health ok, rag_ready true.
Контейнер protocol-gcp-app:a592d588fdd7 running, restart_count 0; started_at
2026-09-06T08:44:05.996563403Z. Это snapshot перед нашей реализацией; он не
подтверждает последующие релизы и не исключает будущие действия другой вкладки.

## Первые PR

| Этап | PR / исходный HEAD | Зона владельца | Локальная проверка | Состояние при записи |
|---|---|---|---|---|
| 1a: calendar API | #205 / 6d9085c3 | mo_backend.py, route-блоки rag_server.py, test_mo_cohort_contract.py | 53 passed, lint/syntax | Опубликован, CI ожидается; не deployed |
| 1b: numerical labs | #206 / 58410dc4 | mo_lab_bundle.py, mo_lab_shadow.py, lab_abnormal_findings.py | 44 passed, lint/syntax | Draft, после 205; не deployed |
| 1c: family integrity | #207 / c48cf62b | mo_finding_families.py, test_mo_family_scores.py | 31 passed, lint | Draft, после 206; не deployed |
| History prior | #208 / 11dfdd72 | mo_history_deep.py, test_mo_history_deep.py | 23 passed, lint | Draft, после 207; не deployed |

В каждом PR свой handoff с ограничениями. Эти counts относятся к разным наборам
с пересечением тестов; не складывать их как число уникальных проверок.
Общий CI, merge SHA и production smoke фиксировать отдельно после завершения.

## Что НЕ закрыто

- A01 частично: основные периоды и facets; полный CohortSpec/hash/все endpoints,
  расширенные фильтры, export parity и сводка №55 ещё требуют реализации.
- A02 исправляет передачу значений. A04 только strict dimensions/numbers и adult
  applicability. Время доступности, локальные референсы, TSH, конверсии и
  клиническая значимость ещё не валидированы.
- A05/A06/A09 улучшены на family contract; полный coverage/status всех evaluators,
  SQL provenance, group denominators, UI unknown ещё не завершены.
- A21 улучшает выбор prior; весь longitudinal context не завершён. Нужны episode
  boundaries, timestamps, действующая терапия и связь history→assessment.
- A03/A07/A08/A10–A20/A22–A32, кроме прямо указанных частичных исправлений,
  не считать закрытыми. Особенно лекарственные negation/form/dose guards,
  №55 evidence, zero readiness, case-mix, human validation и UI.
- Primary flags не включались; production backfill, новые LLM расходы на клинические
  записи и клиническая валидация не выполнялись.

## Порядок продолжения

1. Завершить exact-HEAD CI/review первого PR, merge через GitHub.
2. Синхронизировать следующий task PR с main, сохранив все чужие изменения.
   Опубликованную историю не force-push. Если rebase перепишет опубликованные
   коммиты, использовать новый task branch/PR от main либо безопасную merge-base
   синхронизацию; не обходить запреты reset/clean/force-push.
3. Конфликт одной BUILD_VERSION разрешается штатным helper; реальный конфликт —
   остановить этот перенос и разделить файлы. Проверять фактические hunks:
   #113/#186 меняют rag_server.py только версией. Старый shell wrapper не всегда
   распознаёт version-only peer, но pr_isolation поддерживает этот признак.
4. Актуальные tests/CI после синхронизации; затем следующий merge. Один координатор
   и одно окно GCE release, без параллельной правки env/data/firewall.
5. Release только exact main SHA. Health/version + feature smoke и rollback plan.
6. Следующие независимые работы: zero/readiness, Rceth safeguards, truthful family
   UI, №55/cohort API, клиническая спецификация и эталонная разметка.

## Модель и бюджет

Сильную модель использовать для архитектуры, клинических контрактов и итоговой
проверки; более экономичную можно рассматривать для ограниченных механических
изменений. Модель не заменяет regression tests и врачебную валидацию. Перед
переключением сохранять SHA/PR/handoff; не повторять весь аудит. Смена модели
приложения Protocol не равна смене модели агента-разработчика и требует отдельной оценки.

## Handoff этой документационной задачи

Repo akuazuk/protocol, branch codex/mo-implementation-tracker-agent1-pc1,
worktree /private/tmp/protocol-task-mo-implementation-tracker-pc1,
base a592d588fdd7eb428161024ad13e4e3948bb3754, HEAD — опубликованный commit с этим файлом.
BUILD_VERSION не меняется: только документация и синтетические материалы.
Merge/deploy при первой записи нет. Проверены локальные ссылки, JSON,
синтаксис synthetic UI harness; runtime tests этой задачей не заменяются.
Держим этот tracker, audit и assets/mo-audit-2026-09-06; docs/plans/README.md не трогаем.

Одна безопасная следующая команда:

```bash
gh pr checks 205 --repo akuazuk/protocol
```
