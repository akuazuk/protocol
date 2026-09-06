# MO: CASE Review и аналитика, план v1

Дата: 2026-09-06.
Статус: active.
Предшественники:

- `2026-08-09-mo-case-review-quality-parity-v1.md`;
- `2026-08-05-mo-case-review-workspace-v2.md`;
- `2026-08-08-mo-analytics-ui-target-v2.md`;
- `2026-08-09-mo-dashboards-zones-first-v2.md`;
- `2026-09-04-mo-drugs-labs-scoring-v1.md`.

## Контекст

План объединяет дополнительную проверку CASE Review и всего интерфейса МО
Аналитики. Он продолжает действующие планы, не переписывает их историю и не
считает три отобранных shadow-случая репрезентативной выборкой.

Нельзя утверждать, что evaluator получил пустой вход, пока не проверены вход и
ревизия конкретного запуска. Автоматическое замечание не доказывает клинический
дефект. Новые медицинские пороги, веса, primary-флаги и массовый backfill требуют
отдельного клинического решения.

## Что уже изменено в production

Baseline на момент публикации:

- production SHA: `495b2f439a07a2975c6e27d5115e86a2460401d1`;
- `BUILD_VERSION`: `2026-09-06-155213Z-family-provenance`;
- family group denominators и UI n/N выпущены;
- family finding provenance stage 1 выпущен;
- lab canons входят в GCE image, synthetic image evaluation проходит;
- stale page responses защищены epoch/abort-контрактом;
- primary lab, новые клинические формулы и массовый backfill не включены.

## Метрики и цели

| Метрика | Было | Цель |
|---|---:|---:|
| List/detail/export parity для одной ревизии | не закреплено | 100% synthetic сценариев |
| Synthetic сценарии E | частичное покрытие | 23 из 23 |
| UI-наблюдения U | частично | U01-U14 проверены |
| Viewport matrix | статические проверки | 320/360/768/1024/1440 + zoom 200% |
| Ненулевое значение, показанное как 0% | возможно | 0 |
| Unknown, показанный как success/0% | возможно | 0 |
| Скрытая запись при просмотре/Next | не закреплено | 0 |
| Lab assets в release image | проверено | проверять каждый release |

## Наблюдения R01-R14

| ID | Риск | Требуемый результат |
|---|---|---|
| R01 | Shadow противоречит видимым Dx/code/plan | Проверять canonical input и revision до вывода |
| R02 | Нет КП рядом с verdict о КП | `not_evaluated`, без автоматического нарушения |
| R03 | List и drawer показывают разные N55 | Один persisted run и единое округление |
| R04 | БАК смешан с бакпосевом | Canonical test/order/specimen/method/state |
| R05 | Наличие результата названо игнорированием | Разделить наличие, упоминание, интерпретацию, действие |
| R06 | Any prior смешан с relevant prior | Раздельные признаки и reason codes |
| R07 | Контекст истории попадает в дефекты | Контекст не увеличивает счетчик замечаний |
| R08 | Лекарственный verdict без evidence chain | Факт пациента, назначение, источник нормы, применимость |
| R09 | Балл скрывает неполноту проверки | Coverage и status рядом с value |
| R10 | Post-visit context отделен | Сохранить cutoff и безопасное поведение |
| R11 | Local analog выдан как прямая норма | Явно подписать локальную адаптацию |
| R12 | Итоги и findings дублируются | Один итог и одна карточка на вопрос |
| R13 | Неконкретные формулировки | Показывать применимый отсутствующий факт |
| R14 | Техническое завершение выглядит клиническим | Разделить execution, applicability и verdict |

## Волна A: воспроизводимый assessment

1. Проследить документ, extraction, history/lab bundles, evaluator, persistence,
   SQL projection, list/detail/export.
2. Сохранять immutable metadata запуска:
   `evaluation_run_id`, `document_revision`, `source_hash`, `evaluated_at`,
   `snapshot_at`, `cutoff_at`, `methodology_version`, `evaluator_version`,
   primary/shadow mode, protocol applicability и coverage.
3. Публичный контракт результата:
   `completed|partial|insufficient_data|not_applicable|error|stale|conflict`,
   nullable value, reason codes и evidence refs.
4. Пустой evaluator input при заполненном canonical input означает transport или
   extraction error, а не доказанный дефект.
5. Finding, конфликтующий с evidence текущей ревизии, хранится для аудита, но не
   становится подтвержденным итогом и не попадает в шаблон врачу.
6. List, drawer и export читают один run/revision/snapshot. Округление выполняет
   только presenter.

Приемка: заполненные Dx/code/plan не отображаются отсутствующими из-за пустого
входа; legacy result помечается stale; pending/error не становится success;
list/detail/export совпадают для одной ревизии.

## Волна B: история и лаборатория

1. Разделить `history_available`, `any_prior_exists`,
   `relevant_episode_prior_exists`, `correction_assessable` и причины исключения.
2. Использовать только события до cutoff; same-day evidence без `available_at`
   не считать доказанно доступным.
3. История документа и загруженная longitudinal history остаются разными
   источниками. Нерелевантный prior не делает коррекцию автоматически применимой.
4. Canonical lab identity включает test/order, analyte, specimen, method, state и
   available time. Бакпосев, биохимия, "в работе" и готовый результат не смешивать.
5. Для результата отдельно хранить наличие, упоминание, интерпретацию и действие.
   Без единицы, референса, времени доступности или approved threshold результат
   остается unknown/question.

Приемка: post-visit result не влияет на текущую оценку; БАК не совпадает с
бакпосевом; наличие нормального результата без повторения названия в диагнозе не
создает дефект.

## Волна C: лекарства и нормативная применимость

Каждая карточка содержит назначение, форму, путь, дозу, длительность, статус
активности, assertion/subject/time факта пациента, источник инструкции, редакцию,
применимость и неопределенность.

Отрицание, семейный анамнез и гипотеза не являются подтвержденным
противопоказанием. Альтернативы через "или" не считаются совместным приемом.
Прошлое назначение не считается активным без evidence. Доза одного препарата не
заполняет пропуск другого.

Если протокол не подобран, `protocol_check=not_evaluated`. Нормативную оценку
N55/N127 отделять от локальной методики и проверять ее scope. До clinical gate
результат остается shadow/partial, без новых весов и primary promotion.

## Волна D: CASE Review drawer

1. Заголовок показывает позицию `N из M`, Previous/Next и сохраняет очередь,
   фильтры и сортировку.
2. Desktop имеет документ и проверку в двух колонках; mobile переключает
   "Документ / Проверка" без горизонтального overflow.
3. Справа один итог, затем "Замечания", "История и анализы", "Критерии".
4. Одна evidence-card отвечает на один вопрос: причина, точный span, требование,
   применимость, unknown и решение.
5. Case-bound request identity/AbortController не допускает поздний ответ другого
   случая. Ошибка history/lab не скрывает документ.
6. Просмотр, раскрытие и Next не пишут данные. Draft и saved различаются.
   Unsaved transition требует явного решения пользователя.
7. Save имеет latest-revision check, idempotency, RBAC и audit event.
8. Проверить dialog semantics, focus trap, Esc, focus return, keyboard-only,
   sticky footer и zoom 200%.

## Волна U: весь интерфейс МО Аналитики

| ID | Изменение | Приемка |
|---|---|---|
| U01 | Полные accessible названия навигации | Narrow menu не требует помнить буквы |
| U02 | Viewport-bound filter popover | Панель доступна на всей viewport matrix |
| U03 | Компактная шапка, analysis route только после drill | Пустой rail не занимает высоту |
| U04 | Human-readable cohort chips | Видны все реально действующие ограничения |
| U05 | Один canonical period/filter URL state | URL, controls, API и export совпадают |
| U06 | 6-8 основных колонок и column manager | Нет обязательного 19-column overflow |
| U07 | Нейтральная терминология | Algorithm signal не назван доказанным дефектом |
| U08 | Per-zone assessed denominators | `evaluated_n=0` показывает "Не оценено" |
| U09 | Явные n/N и scope показателей | Group rate не смешан с period contribution |
| U10 | Small-n по evaluated N | Причина suppression видна в строке |
| U11 | Tiny nonzero formatter | `<0,1%` и абсолютное n |
| U12 | Технические family codes только в details | Основной поток на русском |
| U13 | Widget states и bounded retry | Error одного API не очищает экран |
| U14 | Названия по задачам | "Проверка назначений", "Инструкции препаратов" |

Общий analytics contract:

- KPI отвечает: всего, оценено, требует разбора, изменение к сопоставимому периоду;
- unknown в графике является gap, а не нулем;
- одно МО с несколькими findings не увеличивает unique case total;
- drill сохраняет cohort/snapshot, Back возвращает позицию и выделение;
- массовые операции показывают область выбора, preview, partial errors и используют
  idempotency; пустой выбор ничего не записывает;
- report preview показывает cohort, methodology и status; QA только на synthetic.

## Волна E: synthetic acceptance

Постоянный набор содержит 23 сценария:

1. заполненный документ и пустой evaluator input;
2. stale assessment другой revision;
3. гонка Next/Previous;
4. list/detail/export parity;
5. protocol not evaluated;
6. отсутствие history;
7. нерелевантный prior;
8. релевантный prior с used/excluded evidence;
9. бакпосев и биохимия;
10. нормальный lab result без повтора в Dx;
11. отклонение без достаточного контекста;
12. post-visit и unknown same-day time;
13. empty/failed lab evaluator;
14. настоящий нулевой балл;
15. negation/family history/hypothesis;
16. alternative и past medications;
17. consent в отдельном недоступном источнике;
18. local N55 adaptation;
19. suspicion не входит в сообщение врачу;
20. повторный Save, concurrent update и RBAC;
21. 403/500/timeout одного раздела;
22. keyboard/focus/narrow/long text/zoom;
23. lab assets и synthetic evaluation внутри image.

Fixtures только синтетические. Python contract tests и Playwright публикуются
отдельными test-only PR уровня 4.

## Порядок реализации и выпуска

1. Assessment persistence и parity.
2. History/lab identity и safe cutoff.
3. Medication evidence и normative applicability.
4. UI shell U01-U06.
5. Honest analytics U07-U14.
6. Drawer D.
7. Synthetic E и usability.

Каждая волна: новая task-ветка от свежего `origin/main`, overlap check, focused
tests, отдельный test-only PR, `BUILD_VERSION`, required CI, merge, exact-main
GCE release и feature smoke. Runtime/UI и level-4 tests не смешивать.

## Риски и gates

- Clinical gate: новые пороги, веса, primary-флаги и формулы.
- Data gate: backfill, recompute и реальные сообщения врачам отдельным решением.
- Coordination gate: `mo_backend.py`, `mo-app.js`, `rag_server.py`,
  `docs/plans/README.md` меняются последовательно.
- Privacy gate: без PHI в logs, fixtures, PR и handoff.
- Usability gate: пять synthetic задач - выбрать период, объяснить процент,
  открыть evidence, вернуться к срезу, подготовить решение или preview отчета.

## Журнал

Каждый handoff содержит:

`Rxx/Uxx -> PR -> local/CI -> merge SHA -> production SHA -> clinical/usability gate -> остаток`.

Нельзя считать волну завершенной без соответствующей synthetic проверки. Нельзя
объявлять клиническую корректность по техническому CI.
