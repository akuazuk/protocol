# Матрица приёмки A01-A32 / R01-R14 / U01-U14 / E01-E23

Дата старта матрицы: 2026-09-07.
База snapshot: `origin/main` = production `29b8a174` (PR #243 + #245).
Предыдущий snapshot проверки: `a6955ef2` (PR #242).
Этот файл обновляется после каждого этапа remaining-work.
Статусы: `shipped` - в main/проде, но не обязательно клинически принято;
`partial` - код есть, критерий приёмки не закрыт; `open` - подтверждённый остаток;
`blocked` - нужен эксперт/юрист/данные, код не заменяет;
`this_pr` - закрывается текущим PR после merge, в проде ещё нет.

Не складывать green CI разных PR как независимые доказательства.
Не объявлять весь план закрытым без отдельных technical / usability / clinical gates.

## Этап текущей записи

| Поле | Значение |
|---|---|
| Этап | P0-1/P0-2 + P1-1 merged и задеплоены |
| Merge | #243 `46fd75c1`, #245 `29b8a174` (#244 закрыт после rebase) |
| Production | `29b8a174` / `2026-09-07-104339Z-history-same-day` |
| Health | `/health/live` ok; образ `protocol-gcp-app:29b8a1746184` |

## A01-A32

| ID | Требование | Реализация сейчас | Тест | Merge / prod | Остаток |
|---|---|---|---|---|---|
| A01 | Единый cohort/period во всех API | #205 и последующие period/filter PR | `test_mo_cohort_contract` | shipped `a6955ef2` | полный CohortSpec/hash и все endpoints |
| A02 | Lab values не теряются в reconcile | #206/#237 | lab abnormal/bundle | shipped | клиническая значимость единиц |
| A03 | Отрицание не даёт противопоказание | medication/negation guards частичные | family/meds tests | partial | полный assertion/subject/temporality |
| A04 | Строгие единицы и возраст референса | #206 partial | lab unit tests | partial | конверсии, локальные референсы |
| A05 | 100 только после выполненных проверок | family contract #220/#223 | family scores | partial | все evaluators, не только family |
| A06 | Дедуп findings до штрафа | family identity | family scores | partial | primary+shadow promotion path |
| A07 | Dx score не за текст/код | assessment contract #227-229 | wave E / dx tests | partial | клиническая поддержка диагноза |
| A08 | Коррекция плана не за любой prior | history/protocol PR | history tests | partial | same-day + relevance |
| A09 | Честный lab denominator | family/provenance PR | KPI tests | partial | evaluated N везде |
| A10 | Small-n без ложного ранга | honest metrics #233 | UI/API tests | partial | evaluated N врача |
| A11 | Знаменатель группы, не периода | #220/#221/#225 | family tests | partial | review projection |
| A12 | №55 не overwrite чужим cohort | cohort/overview fixes | cohort tests | partial | все фильтры/версия |
| A13 | readiness=0 не теряется | readiness guards | backend tests | partial | UI «недостаточно данных» |
| A14 | Доза привязана к препарату | meds #238/#239 | meds tests | partial | все high-alert пути |
| A15 | Safety action к конкретной угрозе | meds/safety partial | meds tests | partial | срок и конкретный red flag |
| A16 | Назначенное ≠ выполненное | labs/meds lifecycle | lab tests | partial | order/result/interpretation |
| A17 | Rceth продукт/форма/дата | rceth path partial | label tests | partial | неоднозначность → unknown |
| A18 | Детский подраздел ≠ запрет взрослым | posology partial | rceth tests | partial | разбор условий разделов |
| A19 | Off-label ≠ автодефект | indication graph нет | - | open | indication graph |
| A20 | Same-day history + episode graph | #245 same-day по timestamp | `test_mo_history_same_day` | shipped `29b8a174` | episode graph, timezone канон склада, cutoff в case detail API |
| A21 | Relevant prior, не richest | history deep #208/#236 | history deep | partial | unrelated-rich prior regression |
| A22 | Сбой evaluator ≠ «без нарушений» | coverage/status partial | wave E E13 | partial | все модули |
| A23 | Family KPI по provenance | #223/#225 | family tests | partial | confirmed/rejected projection |
| A24 | Убрать псевдо-надёжность UI | honest metrics #233 | UI tests | partial | estimated flag в KPI |
| A25 | Completeness guard, plan=na виден | assessment contract | wave E | partial | grade vs unknown plan |
| A26 | Case-mix out-of-time | нет | - | open | отдельный PR |
| A27 | Keyboard drill в таблицах | shell #231 | zoom/keyboard E22 | partial | doctor/specialty drill |
| A28 | Abort устаревших ответов | #218/#219 | stale tests | shipped | гонка смены случая в drawer |
| A29 | MO E2E pack | Wave E #242 | `test_mo_wave_e_acceptance` | partial | E04/E23 не поведенческие |
| A30 | Eligibility отдельно от verdict | #243 opt-in, revoke, export filter | `test_mo_review_pack_concurrency` | shipped `29b8a174` | holdout/split; HTTP revoke; юр. review |
| A31 | Один канон GCE/docs | AGENTS + runbook | hygiene | partial | stale Render mentions в старых docs |
| A32 | Матрица охвата услуг | UI scope partial | - | open | неclinical vs clinical явно |

## R01-R14

| ID | Требование | Реализация | Тест | Merge / prod | Остаток |
|---|---|---|---|---|---|
| R01 | Shadow не противоречит input | assessment/input #227-229 | wave E | shipped | клиническая проверка |
| R02 | Нет КП → not_evaluated | protocol gate #230/#239 | E05 | shipped | применимость экспертом |
| R03 | List/drawer один N55 | assessment contract | E04 signature | partial | HTTP list/detail/export |
| R04 | БАК ≠ бакпосев | lab identity #237 | E09 | shipped | method/specimen полнота |
| R05 | Наличие ≠ игнорирование | lab lifecycle | E10/E11 | partial | действие врача |
| R06 | Any prior ≠ relevant | history assessment | E07/E08 | partial | same-day cutoff |
| R07 | Контекст истории не в дефекты | history flags | history tests | partial | query_failed status |
| R08 | Drug evidence chain | meds cards #238 | E16 | partial | каждый вариант альтернативы |
| R09 | Coverage рядом с баллом | honest metrics | E14 | partial | evaluated N |
| R10 | Post-visit cutoff | history/lab cutoff | E12 | partial | unknown time status |
| R11 | Local analog подписан | N55 adaptation | E18 | shipped | юридическая приёмка |
| R12 | Один итог / одна карточка | drawer #234 | UI tests | partial | дубли в analytics |
| R13 | Конкретный отсутствующий факт | findings cards | wave E | partial | формулировки экспертом |
| R14 | Execution ≠ clinical verdict | assessment status | E14/E19 | partial | analytics↔methodist |

## U01-U14

| ID | Требование | Реализация | Тест | Merge / prod | Остаток |
|---|---|---|---|---|---|
| U01 | Полные названия меню | #231, видно на prod | browser snapshot 07.09 | shipped | - |
| U02 | Filter popover в viewport | #231 | E22 partial | shipped | zoom отдельно |
| U03 | Компактная шапка | #231 | browser snapshot | shipped | - |
| U04 | Human-readable chips | #232 | browser snapshot | shipped | абсолютные даты периода |
| U05 | Canonical URL state | filters #232 | cohort tests | partial | два поиска |
| U06 | 6-8 колонок + manager | #232 | browser | partial | filter row vs hidden cols |
| U07 | Нейтральная терминология | metrics #233 | browser | open | «Shadow: плохо/критично» |
| U08 | Per-zone assessed N | honest metrics | API | partial | evaluated_cases=None |
| U09 | Явные n/N и scope | #233 | API/UI | partial | review projection |
| U10 | Small-n по evaluated N | #233 | API | partial | evaluated N |
| U11 | Tiny nonzero formatter | #233 | UI | shipped | - |
| U12 | Family codes в details | #233 | UI | shipped | - |
| U13 | Widget states / retry | #240/#241 | E21 | partial | 409 compare UI |
| U14 | Названия по задачам | #231 | browser | open | дубль «Справка» |

UX-остаток P1-4 на prod кадре 2026-09-07: Shadow-подпись, два поиска, период
«текущий месяц» без дат, bulk actions при n=0, filter row шире видимых колонок,
дубль Справки. Не закрывать U* только номером PR.

## E01-E23

| ID | Сценарий | Тест сейчас | Исполняемое действие | Prod | Остаток |
|---|---|---|---|---|---|
| E01 | evaluator input loss | wave E | unit contract | shipped | - |
| E02 | stale revision | wave E + stale cohort | unit | shipped | drawer race |
| E03 | next/previous race | wave E | unit | shipped | - |
| E04 | list/detail/export parity | 3× `_assessment_contract_from_row` | нет HTTP | open | HTTP одной synthetic БД |
| E05 | protocol not evaluated | wave E | unit | shipped | clinical gate |
| E06 | history absent | wave E + #245 query_failed | unit | shipped `29b8a174` | - |
| E07 | irrelevant prior | wave E | unit | shipped | same-day |
| E08 | relevant prior evidence | wave E | unit | shipped | timestamps |
| E09 | culture vs chemistry | wave E | unit | shipped | - |
| E10 | normal lab | wave E | unit | shipped | - |
| E11 | abnormal without context | wave E | unit | shipped | - |
| E12 | post-visit / unknown time | wave E | unit | shipped | unknown status |
| E13 | empty/failed lab | wave E | unit | shipped | - |
| E14 | true zero | wave E | unit | shipped | - |
| E15 | negation family | wave E | unit | shipped | A03 полнота |
| E16 | alternative/past meds | wave E | unit | shipped | all alternatives |
| E17 | consent/unavailable | wave E | unit | shipped | - |
| E18 | local N55 | wave E | unit | shipped | - |
| E19 | suspicion not to doctor | wave E | unit | shipped | - |
| E20 | save / replay / RBAC | #243 HTTP/SQLite concurrency | concurrent sessions, hash, 403/409 | shipped `29b8a174` | side-by-side compare UI |
| E21 | isolated widget fail | wave E | unit | shipped | - |
| E22 | keyboard / narrow / zoom | viewport reflow | Playwright partial | partial | настоящий browser zoom |
| E23 | lab assets in image | GCE build 29b8a174 ran verifier | image step ok ranges=8 panels=17 | partial | постоянный CI job, не только deploy log |

## Критерии этого этапа

P0-1: два клиента, первый save побеждает, второй 409, текст первого цел;
одинаковый POST = один pack; другой payload того же key = конфликт; чужой
run/pack отвергается; viewer 403, methodist 200. Синтетика, без prod mutation.

P0-2: явный false любой роли сохраняется; отсутствие eligibility не экспортируется;
отзыв исключает будущую выборку. Holdout/split и юр. вывод - отдельно.

## Следующие этапы (новые worktree, не эта ветка)

1. P1-2 evaluated denominators + methodist review projection (`mo_backend.py`).
2. P1-3 E04 HTTP list/detail/export parity; E23 как постоянный CI.
3. P1-4 UX leftover (Shadow-подпись, два поиска, даты периода, bulk n=0, Справка).
4. P1-5 clinical/normative gates - не код.
