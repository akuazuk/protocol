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

## Checkpoint реализации: 2026-09-06, после совместной проверки

#205 merged: fe0734a8a5956d1e7a8d494da895319411968d01. Все required PR checks
успешны. Main CI на момент checkpoint ещё выполняется. Production нами не менялся.
#206 обновлён merge main без force-push: 82ff5df4; 49 локальных тестов вместе
с календарным API passed; новый CI ожидается, auto-merge включён после gates.

Дополнительные опубликованные работы:

| PR | HEAD при checkpoint | Изменение | Проверка |
|---|---|---|---|
| #211 | 2758c0cd | Unknown admission/matrix и сохранение readiness=0 | 23 passed |
| #212 | 07d35754 | Видимые unknown/partial, отказ от выдуманной уверенности, честная подпись процентов семейств, keyboard drill | 41 passed; 4 browser payload, Enter, mobile |
| #213 | d59d87f5 | Guard отрицаний/семейных фраз, возраст детского подраздела, текстовый finding как candidate | 28 passed |
| #214 | f05b2e3a | Единые GCE/merge/worktree инструкции | docs links/bash syntax/diff |

#207/#208/#211–214 пока draft, не merged и не deployed. Точные актуальные SHA
проверять в GitHub: checkpoint не заменяет текущее состояние. Primary flags
и persisted оценки не изменялись. #212 исправляет подпись фактического общего
знаменателя, но не добавляет group-specific rate. #213 — ограниченный guard,
не полный assertion/subject/temporality parser; clinical review остаётся.

Совместная проверка #205–208/#211–213: 174 passed (19 тематических файлов),
browser 4 payload, zero/unknown, parse_ok без ложных 90%, mobile 318/318,
keyboard Enter → documents + нужный finding_codes. Интеграционная ветка
codex/mo-integration-verification-agent1-pc1, HEAD e69d16f419768b9c2995745571a58c19f8ccb41d,
worktree /Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/mo-integration-verification.
Это локальная диагностическая сборка, НЕ release и НЕ интеграционный PR для merge.
Код доставляется отдельными PR; не деплоить эту ветку.

### Сохранность и Cursor

В ходе работы исчез временный worktree dual-score; причина не установлена.
Незакоммиченный небольшой diff восстановлен, повторно 23 passed, опубликован #211.
Новые worktree созданы в постоянном Protocol-worktrees, все активные защищены lock.
Не удалять, не unlock и не prune чужую текущую работу при cleanup.
Инструкция размещена в комментарии к #210:
https://github.com/akuazuk/protocol/pull/210#issuecomment-5558200178

Следующий release-координатор должен проверить текущий auto-merge #206 и
завершение CI. До собственного окна GCE исключить параллельные auto-merges
runtime PR. Последовательно обновлять дальнейшие ветки от merged main,
разрешая только BUILD_VERSION штатным helper; проверять актуальный HEAD CI.
Актуальный GCE runbook и daily checklist согласованы в #214, пока не merged.

## Финальная передача Cursor по просьбе пользователя

Этот checkpoint заменяет оперативные статусы предыдущего checkpoint выше.
[Начать здесь: подробный handoff](2026-09-06-handoff-cursor-mo-continuation.md).
[Клинические решения и критерии допуска](2026-09-06-mo-clinical-review-gates.md).

Реализация остановлена для передачи: #205/#206 merged, main e15ac9cf с успешным CI.
Production остаётся a592d588; наших deploy не было. Обнаружен блокер упаковки:
в работающем image отсутствует lab_reference_ranges.json, каталог lab_canons
не включён в archive/COPY. Исправление ещё не написано; выполнить первым.
#207 возвращён в draft, все наши открытые runtime PR без auto-merge.
У #212 на последнее чтение нет CI текущего HEAD: не считать его green.
#215 содержит 14 успешных локальных browser checks и успешный CI.
Cursor ведёт #216 (dead-branch guard); сохранить его и согласовать #214.
Полные SHA, PR, worktree, ограничения и последовательность — в handoff выше.
Не запускать новые изменения или релиз из этой документационной задачи.

## Checkpoint продолжения: 2026-09-06, runtime batch выпущен

Исторические статусы выше сохранены как журнал. Актуальное состояние:

| Требование / этап | Реализация | Проверка | Merge SHA | Production |
|---|---|---|---|---|
| A01, единый календарный срез | #205 | 53 focused; required CI и main CI | `fe0734a8` | `246e3533` |
| A02/A04, числовой lab context | #206 + packaging #217 | 49 focused; image build; running-image synthetic evaluation: ranges=8, panels=17, shadow=1 | `e15ac9c`, `ce124215` | `246e3533` |
| A05/A06/A09, family uncertainty/dedupe | #207 | 58 focused после sync; required CI | `b1b4416b` | `246e3533` |
| A21, релевантный prior | #208 | 23 focused; required CI | `361dc4b2` | `246e3533` |
| A13/A23, unknown dual scores | #211 | 23 focused; required CI | `0552fc35` | `246e3533` |
| A03/A18/A19, Rceth assertion guards | #213 | 28 focused; required CI | `7fee1f7b` | `246e3533` |
| A24/A25/A27, видимые unknown и keyboard drill | #212 | 41 focused + browser; required CI | `13110077` | `246e3533` |
| A29, постоянный MO browser acceptance | #215 | 20 E2E на финальном main; required CI | `246e3533` | `246e3533` |
| A31, единый GCE/worktree канон + guard #216 | #214 + #216 | 23 docs/guard tests; bash syntax; required CI | `e7121a69`, `8270b874` | docs merge `e7121a69` ещё не deployed |

Точный production:

- `git_commit=246e35336f8b73b3f66e31c38b0d58506c1ce099`;
- `version=2026-09-06-115045Z-mo-score-availability-ui`;
- `/health/live`, MO health/capabilities, search, CSP/HSTS и CORS smoke успешны;
- image `protocol-gcp-app:246e35336f8b`, bind `127.0.0.1:8000`;
- оба `data/lab_canons/*.json` присутствуют в работающем контейнере.

Первый corpus-sync оборвался по SSH после очистки удалённого summary-каталога.
До продолжения deploy корпус восстановлен каноническим sync: 478 PDF и 467
summary. Затем release завершён с `SYNC_PROTOCOL_CORPUS=0`; production corpus
и `rag_ready=true` повторно проверены.

Это не закрывает весь A01-A32. Остаются, в частности: полный CohortSpec/hash и
parity всех endpoint/export (A01/A12/A28), lifecycle/available_at и статусы
lab checks (A16/A22), provenance и групповые знаменатели (A10/A11/A23),
утверждённый №55 mapping (A07/A08/A15), полная longitudinal medication timeline
(A14/A17/A20/A21), case-mix (A26), training eligibility (A30) и scope услуг
(A32). Новые clinical weights и primary flags не включены; клинический gate
не пройден.

Следующий независимый инженерный этап без изменения клинической методики:
A28 - отмена старых запросов и защита UI от перерисовки устаревшим cohort.
