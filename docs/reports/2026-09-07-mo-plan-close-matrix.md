# Матрица закрытия remaining-work МО - 2026-09-07

Владелец закрыл remaining-work план 2026-09-07 с явной оговоркой:
технические пункты этого этапа выпущены в коде; клинические и нормативные
гейты не заменялись экспертизой. Это не научная валидация весов, протоколов
или №55.

Статусы: `shipped` - код в main/проде после этого релиза;
`owner_deferred` - владелец снял блок плана, критерий экспертизы не выполнен;
`partial` - код есть, узкий остаток не клинический.

Не редактирует `2026-09-07-mo-remaining-acceptance-matrix.md` (файл #246).

## Snapshot на момент закрытия плана

| Поле | Значение |
|---|---|
| Base до этого PR | `29b8a174` / `2026-09-07-104339Z-history-same-day` |
| Уже в main | #217-#245 |
| Этот этап | P1-2 evaluated N, P1-3 E04/E23, P1-4 UX |
| P0 / P1-1 | #243 / #245, уже в проде |

## A01-A32

| ID | Статус закрытия | Доказательство | Оговорка владельца |
|---|---|---|---|
| A01 | shipped | cohort HTTP tests | полный CohortSpec hash не расширялся |
| A02 | shipped | lab lifecycle #237 | клиническая значимость единиц - owner_deferred |
| A03 | owner_deferred | negation guards частичные | полный assertion/temporality не сертифицирован |
| A04 | owner_deferred | unit guards | локальные референсы и конверсии |
| A05 | partial | family contract | не все evaluators |
| A06 | partial | family identity | primary+shadow promotion |
| A07 | owner_deferred | assessment contract | клиническая поддержка Dx |
| A08 | shipped | same-day #245 | relevance остаётся правилами кода |
| A09 | shipped | evaluated N this PR | только family KPI этого контура |
| A10 | shipped | ranking по evaluated N >= 20 | - |
| A11 | shipped | group + evaluated denominators | - |
| A12 | partial | cohort tests | все фильтры/версии |
| A13 | partial | readiness guards | - |
| A14 | owner_deferred | meds cards | все high-alert пути |
| A15 | owner_deferred | safety cards | срок red flag |
| A16 | partial | lab lifecycle | order/result/interpretation |
| A17 | owner_deferred | rceth path | неоднозначность |
| A18 | owner_deferred | posology | разделы инструкции |
| A19 | owner_deferred | indication graph нет | отдельная клиническая задача |
| A20 | shipped | same-day #245 | timezone канон склада не добавлялся |
| A21 | partial | history deep | unrelated-rich regression |
| A22 | partial | E13 | все модули |
| A23 | shipped | review projection this PR | latest CRM only, не shadow=human |
| A24 | shipped | honest metrics + UX this PR | - |
| A25 | partial | assessment status | - |
| A26 | owner_deferred | case-mix нет | отдельный PR |
| A27 | partial | keyboard E22 | doctor drill |
| A28 | shipped | stale abort #218/#219 | - |
| A29 | shipped | Wave E + HTTP E04 + verifier E23 | не built-image CI job |
| A30 | shipped | #243 opt-in/revoke | holdout/split и юр. review - owner_deferred |
| A31 | partial | GCE канон | stale Render в старых docs |
| A32 | owner_deferred | scope chips | полный каталог услуг |

## R01-R14

| ID | Статус | Доказательство | Оговорка |
|---|---|---|---|
| R01 | shipped | assessment/input | клиническая проверка owner_deferred |
| R02 | shipped | protocol not_evaluated | применимость экспертом owner_deferred |
| R03 | shipped | E04 HTTP list/detail/export | одна synthetic БД |
| R04 | shipped | E09 | method/specimen полнота owner_deferred |
| R05 | partial | lab lifecycle | действие врача |
| R06 | shipped | same-day + relevant flags | - |
| R07 | shipped | query_failed #245 | - |
| R08 | owner_deferred | meds cards | каждый вариант альтернативы |
| R09 | shipped | evaluated N this PR | - |
| R10 | shipped | cutoff #245 | - |
| R11 | shipped | local N55 label | юридическая приёмка owner_deferred |
| R12 | partial | drawer | дубли analytics |
| R13 | owner_deferred | findings cards | формулировки экспертом |
| R14 | shipped | review projection this PR | latest CRM, не все superseded цепочки |

## U01-U14

| ID | Статус | Доказательство |
|---|---|---|
| U01 | shipped | полные названия меню |
| U02 | shipped | filter popover |
| U03 | shipped | компактная шапка |
| U04 | shipped | chips + абсолютные даты периода this PR |
| U05 | shipped | поиск выборки vs фильтр таблицы разделены |
| U06 | shipped | column visibility после bindSortableHeaders |
| U07 | shipped | «Автоматические сигналы: требуют проверки» |
| U08 | shipped | evaluated_cases в family rows |
| U09 | shipped | n/N evaluated + review counts |
| U10 | shipped | small-n по evaluated N |
| U11 | shipped | tiny nonzero |
| U12 | shipped | family codes |
| U13 | shipped | 409 сохраняет черновик #243 |
| U14 | shipped | дубль «Справка» убран из nav |

Независимый usability walkthrough пяти задач на живом проде не проводился.
Владелец принял техническое закрытие U*.

## E01-E23

| ID | Статус | Исполняемое действие |
|---|---|---|
| E01-E03 | shipped | wave E unit |
| E04 | shipped | HTTP list/detail/export одной synthetic БД |
| E05-E19 | shipped | wave E unit |
| E20 | shipped | #243 HTTP/SQLite concurrent + RBAC + hash |
| E21 | shipped | wave E unit |
| E22 | partial | viewport/keyboard; настоящий browser zoom owner_deferred |
| E23 | shipped | `verify_lab_assets.main()` исполняется в pytest; GCE image verifier остаётся в deploy |

## Что код этого PR не закрывает

- Blind clinical set, независимая разметка, разбор разногласий.
- Indication graph (A19), case-mix (A26), patient/time holdout.
- Юридический review eligibility обучения.
- Настоящий browser zoom и полный desktop/mobile usability sign-off.
- Built-image CI job (мешает занятый `ci.yml` dependabot #199); verifier
  исполняется в pytest и в GCE deploy image.

Эти пункты сняты с плана решением владельца 2026-09-07, не доказательством.
