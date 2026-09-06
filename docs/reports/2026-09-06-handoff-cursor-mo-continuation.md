# СТАРТ ДЛЯ CURSOR: продолжение МО без повторного аудита и потери изменений

Дата передачи: 2026-09-06. Причина остановки: пользователь сообщил об исчерпании
кредитов и попросил передать оставшуюся реализацию Cursor. Новую функциональную
работу и релизы предыдущий исполнитель прекратил. Это передача работы, не
заявление о завершении всего плана.

## 1. Что прочитать и в каком порядке

1. Актуальный AGENTS.md из репозитория и .cursor/rules/repository-coordination.mdc.
2. Этот документ целиком — оперативное состояние и порядок продолжения.
3. [Исходный подробный аудит и roadmap](2026-09-06-mo-comprehensive-audit-and-roadmap.md):
   32 находки A01–A32, архитектура, все основные оценки, UI inventory, этапы 0–6,
   критерии приёмки и медицинские ограничения. Это основной scope задания.
4. [Tracker реализации](2026-09-06-mo-implementation-tracker.md) и handoff каждого PR.
5. [Конкретные клинические решения и gates](2026-09-06-mo-clinical-review-gates.md).
6. Актуальный план из docs/plans/README.md, workflow v3, GCE runbook и свежий
   handoff Cursor из #210. Проверить новый #216 прежде, чем менять Git tooling.

Не повторять весь аудит с нуля. Сначала подтвердить изменившееся состояние,
закончить уже подготовленные PR, затем выполнять оставшиеся пункты roadmap.
Снимки этого документа устаревают: GitHub и live GCE проверяются заново.

## 2. Подтверждённое состояние на передачу

- Repo: https://github.com/akuazuk/protocol.git.
- Свежий origin/main: e15ac9cfceac46e9eed51efb65ab3850390a99e1.
- В main уже вошли #205 (календарные фильтры) и #206 (числовой lab context).
- CI main e15ac9cf: success, run 34024408570. Предыдущий main fe0734a8 тоже green.
- Production: https://protocol.kravira.by, GCE project protocol-home-e1,
  VM protocol-app, zone europe-central2-a.
- Live /api/version: git_commit a592d588fdd7eb428161024ad13e4e3948bb3754,
  version 2026-09-06-073651Z-deploy-lock-allowlist, rag_ready=true.
- /health/live: ok=true. Container protocol-gcp-app:a592d588fdd7, running,
  restart_count=0. При последнем чтении процессов docker build не обнаружено.
- НИ ОДИН наш новый runtime PR ещё не развёрнут на GCE. Deploy-скрипт не запускался.
- Primary flags, production backfill и платные клинические LLM-прогоны не запускались.
- Наши открытые runtime PR имеют auto-merge=false. #207 возвращён в draft
  специально для передачи. Чужие #210/#216 имели auto-merge=true: не считать
  main замороженным и согласовать release window с текущим владельцем Cursor.

Начальный checkout Cursor был грязным и на старой ветке. Предыдущий исполнитель
его не правил. При финальном чтении Cursor уже перевёл корневой checkout на
main e15ac9cf. Не принимать ранние заметки о грязной ветке за текущее состояние;
выполнить git status заново. Не писать код непосредственно в этом main.

## 3. Главный блокер следующего релиза — СНАЧАЛА исправить упаковку

Подтверждено read-only внутри работающего контейнера:
/app/data/lab_canons/lab_reference_ranges.json отсутствует (lab_reference_missing).
В deploy/gcp-app/deploy_to_gce.sh git archive allowlist не включает data/lab_canons;
в deploy/gcp-app/Dockerfile нет COPY этого каталога.

Нужны оба исходных файла:

- data/lab_canons/lab_reference_ranges.json — читает lab_abnormal_findings.py;
- data/lab_canons/lab_test_canons.json — читает lab_canons.py.

Поэтому локальный green pytest не доказывает работу lab-проверки в GCE-образе.
Не считать #206 полностью выпущенным только по health/version. Исправление
упаковки ещё НЕ написано, НЕ закоммичено, новый PR под него НЕ создан.
Номер #216 занят другой работой Cursor, не называть его lab packaging PR.

Порядок:

1. Проверить актуальные PR по Dockerfile и deploy_to_gce.sh. #195 меняет тот же
   Dockerfile, предлагая Python 3.14; на передачу docker-images и hygiene красные.
   Не мержить Python upgrade ради освобождения файла, не закрывать чужой PR
   молча. Согласовать владение/последовательность по workflow v3; это реальное
   пересечение файла, а не только BUILD_VERSION.
2. В новой task-ветке от свежего main добавить необходимые lab assets в git
   archive allowlist и Docker COPY. Содержимое клинических справочников в этой
   задаче не менять. Не заменять это ручным копированием на production.
3. Проверить остальные файловые зависимости изменённых evaluators: наличие
   в git не означает наличие в image. Сравнить manifest архива и файлов образа.
4. Добавить проверку состава release artifact: JSON читаются из собранного образа,
   load_reference_ranges/lab_panels возвращают непустой результат. Проверить
   synthetic числовой анализ через evaluate_lab_for_case в image с in-memory SQLite.
5. Поднять BUILD_VERSION, релевантные tests/lint, полный required CI, review,
   merge через GitHub. Не ослаблять checks ради упаковки.
6. Только после этого возможен релиз lab-кода. Проверить runtime dependencies
   внутри точного SHA-образа и провести feature smoke из раздела 7.

## 4. Уже подготовленные PR — использовать, не переписывать заново

Все HEAD ниже — snapshot. Перед каждой работой запросить актуальный SHA.

| PR | HEAD / merge | Содержание | Проверки и состояние |
|---|---|---|---|
| #205 | merge fe0734a8a5956d1e7a8d494da895319411968d01 | period/month в cases/facets/overview/drugs-labs API, календарный срез | 53 local passed, required CI passed, MERGED, не deployed |
| #206 | merge e15ac9cfceac46e9eed51efb65ab3850390a99e1; branch 82ff5df4088c0fab642c306c4b0de54bb9bb87cb | Полный числовой lab bundle отдельно от display/reconcile, cutoff по дате, strict units/numbers, adult seed applicability | 44 первоначально; 49 после sync; CI passed, MERGED; packaging block выше |
| #207 | b2d8c33948c5f32461c7d0d3e440fd42a07f414b | Unknown вместо пустых 100, dedupe, запрет shadow fallback в primary, честный lab denominator | 31 первоначально; 58 после sync e15ac9cf; required green; DRAFT, auto OFF |
| #208 | 11dfdd72c1e6fe485c9afec99649041834f7b23b | Релевантный prior эпизода и свежесть вместо самого полного чужого визита | 23 passed, required green на старой базе; DRAFT |
| #209 | branch codex/mo-implementation-tracker-agent1-pc1 | Полный аудит, tracker, artifacts и эта передача | Только docs; финальный HEAD — commit с этим handoff; новый CI после публикации проверять отдельно |
| #211 | 2758c0cdb3e6fee55426c83bb9b028dcacde7da0 | Настоящий readiness=0 сохраняется; missing component → admission null, matrix unknown | 23 passed; required green на старой базе; DRAFT |
| #212 | 07d35754463d18c2daee1cfe18c45a1ff664031d | Видимые unknown/partial, нет выдуманных 90/75/55% уверенности, подписи family denominator, keyboard drill | 41 passed + browser; DRAFT. На финальное чтение statusCheckRollup=[] и gh run list пуст: выяснить и запустить полный CI, НЕ считать green |
| #213 | d59d87f55b061a2633730a505d2556b12e50ec00 | Детская дозировка не задаёт max-age препарата; отрицания/семейные фразы; текстовые findings как candidates | 28 passed, required green на старой базе; DRAFT |
| #214 | f05b2e3ae85e40247c99a876843a4f28acf11163 | Устранены противоречия GCE/Render/reset, правила сохранности worktree и published merge | Docs checks и required green; DRAFT |
| #215 | df5d928f808f5ae471e2fa7590b476392cfc1734 | Постоянный Playwright MO smoke, настоящий HTML/CSP/ECharts, synthetic API | 14 local browser passed; required green; DRAFT |

Совместная диагностическая сборка #205–208/#211–213:
174 passed в 19 тематических Python-файлах; browser: 4 состояния, ноль/unknown,
parse_ok без фиктивной уверенности, mobile width=scrollWidth=318, Enter drill
с finding_codes. Это не врачебная валидация и не production smoke. Числа отдельных
pytest-наборов пересекаются — не складывать их как уникальные тесты.

## 5. Новые действия Cursor, которые обязательно сохранить

#216, cursor/dead-branch-guard-agent1-pc1, HEAD
4c06d2a02e55adf93534d6d924f3f38fb00ceaf4: страж мёртвых веток, pre-commit,
check_branch_alive.py, подключение hooks, исправленная GCE-подсказка task_start.
На snapshot CI ещё выполнялся, auto-merge=true. Он намеренно НЕ меняет AGENTS.md
и workflow v3, которыми владеет #214. После #216 дополнить #214 формулировкой
проверки живой ветки; не удалять и не дублировать guard. Не использовать
ALLOW_DEAD_BRANCH или --no-verify для обхода реального запрета.

#210 — handoff Cursor по релизу и triage. Исходные #148 (history billed key),
#158 (MIS только GCE), #204 (requirements-rag.lock в deploy) уже включены в main.
Сохранить их изменения. Render приостановлен, его старые runbook не выполнять.

## 6. Владение worktree и безопасная последовательность merges

После завершения этой передачи новый единственный агент Cursor может принять
владение нашими НЕmerged задачами. Не запускать двух исполнителей на них сразу.
Имена опубликованных codex-веток сохранять; переименование не требуется.

| Работа | Worktree |
|---|---|
| #207 | /private/tmp/protocol-task-mo-family-score-integrity-pc1 |
| #208 | /private/tmp/protocol-task-mo-history-prior-selection-pc1 |
| #209 | /private/tmp/protocol-task-mo-implementation-tracker-pc1 |
| #211 | /Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/mo-dual-score-unknown |
| #212 | /Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/mo-score-availability-ui |
| #213 | /Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/mo-label-assertion-guards |
| #214 | /Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/gce-coordination-docs |
| #215 | /Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/mo-browser-acceptance |

Все активные worktree locked. Во время работы один временный worktree исчез;
причина не установлена. Малый diff восстановлен и опубликован #211. Не удалять
и не unlock чужие каталоги при cleanup; не делать массовый prune/clean/reset.
Перед передачей не осталось незакоммиченного runtime-кода этих задач.

Диагностический worktree mo-integration-verification, HEAD
 e69d16f419768b9c2995745571a58c19f8ccb41d — только локальная совместная проверка,
НЕ merge/release branch. Не отправлять его в main вместо отдельных PR.
Release worktree mo-release-e15ac9cf — detached e15ac9cf; деплой не начат.
В нём ignored .env symlink на исходный локальный .env; не читать/печатать секреты.
Если main изменился, сделать новый release worktree, не reset этот каталог.
Merged ветки #205/#206 не использовать для новой работы.

Предпочтительный порядок после устранения packaging block:
#207 → #208 → #211 → #213 → #212 → #215. #209/#214 — документация,
синхронизировать отдельно с учётом #210/#216. UI #212 понимает старый payload,
но принимать окончательно вместе с family #207 и unknown #211.

Перед каждой следующей веткой:

```bash
git status --short --branch
git fetch --prune origin
git rev-list --left-right --count origin/main...HEAD
gh pr list --repo akuazuk/protocol --state open
python3 scripts/ops/pr_dashboard.py --files ПУТИ_ТЕКУЩЕЙ_ЗАДАЧИ
```

Проверить PR state и branch alive, если guard уже merged. Если это чужой,
грязный или уже merged checkout — не чинить его, создать новый clean task worktree.
Перед sync именно принятой опубликованной task-ветки убедиться, что её commit
и remote HEAD совпадают, worktree чистый, другой владелец не пишет параллельно.

Опубликованную историю не переписывать. Допустим merge свежего origin/main
в принятую живую task-ветку без force-push. Для неопубликованных коммитов —
штатный rebase helper. Альтернатива — новая ветка/PR от свежего main с переносом
своих изменений и явным закрытием заменённого PR после проверки.

Конфликт только BUILD_VERSION разрешается scripts/ops/pr_isolation.py
resolve-rag-server с base/ours/theirs из git index; затем bump_build_version.sh.
Helper отказывает при реальном конфликте. Не брать whole-file ours/theirs:
так можно потерять API #205. Любой иной конфликт — остановить этот перенос,
согласовать владельцев/разделение файлов, продолжить независимую работу.

После sync: релевантные tests, syntax/lint, git diff --check, commit, обычный push,
CI НОВОГО HEAD, review, снятие draft и merge через GitHub. Не обходить red CI.
Строгий required CI занимает примерно 12–20 минут; новый main может требовать
повторного прогона. Не пушить каждую мелкую строку отдельно: собрать проверенный
этап, чтобы не плодить очереди. Не отменять чужие/идущие checks и не ослаблять gates.

## 7. Приёмка и выпуск каждого технического этапа

Локально доступны /opt/homebrew/bin/pytest (Python 3.11) и isolated ruff
/private/tmp/protocol-mo-check-tools/bin/ruff. Обычный python3 на Mac — 3.14,
в нём может не быть pytest. На другом компьютере использовать проектный venv
и pinned requirements, не менять зависимости проекта ради запуска проверок.

Наборы по темам:

- Calendar: test_mo_cohort_contract, test_mo_backend, test_mo_month_report,
  test_mo_metrics, test_mo_meds_labs_dashboards.
- Labs: test_mo_lab_clinical_context, test_mo_lab_bundle, test_mo_lab_shadow,
  test_lab_abnormal_and_formulary; упаковка image — отдельная обязательная проверка.
- Family: test_mo_family_scores, test_mo_finding_families, test_kz_deep_eval,
  test_mo_meds_labs_dashboards, затем cohort/lab после sync.
- History: test_mo_history_deep, test_mo_history_continuity,
  test_mo_patient_history_bundle; сохранить тесты billed key #148.
- Dual: test_mo_dual_score_unknown, test_mo_drugs_labs_wave4, test_kz_deep_eval.
- Drugs: test_rceth_label_findings, test_rceth_sync, test_mo_family_scores.
- UI: test_mo_frontend_structure, test_mo_ui_phase2, test_frontend_escaping_guard,
  test_mo_meds_labs_dashboards; node --check frontend/web/shared/mo-app.js.
- Browser: npm ci, npx playwright install chromium, npx playwright test.
  В #215 новый tests/e2e/mo-smoke.spec.ts обнаруживается автоматически.

Пример Python-команды (имена файлов уточнить по checkout):

```bash
/opt/homebrew/bin/pytest tests/test_mo_family_scores.py tests/test_mo_finding_families.py tests/test_kz_deep_eval.py -o addopts='' -q
git diff --check
```

Дополнительный browser harness #212:
docs/reports/assets/mo-score-availability-ui/browser-check.cjs.
Он использует локальный Chrome и Playwright на NODE_PATH; не предназначен для
реальных клинических данных. Постоянный spec #215 проверяет также настоящий CSP.

Release gate:

1. Все нужные изменения в main, required checks на актуальном PR HEAD успешны,
   CI выбранного main SHA успешен; никакого параллельного release/auto-merge
   runtime PR. Согласовать окно с Cursor, который сейчас ведёт #216.
2. Новый чистый release worktree ровно свежего origin/main; BUILD_VERSION
   соответствует коммиту. Проверить конфигурацию, сохранить rollback SHA.
3. По каноническому GCE runbook: bash deploy/gcp-app/deploy_to_gce.sh.
   SYNC_PROTOCOL_CORPUS=0 допустим для сохранения уже развёрнутого корпуса;
   не трогать MIS cron/env/firewall/данные и scoring flags попутно.
4. Проверить публичные /health/live и /api/version: version + git_commit,
   затем image tag, restart status и наличие обоих lab JSON внутри контейнера.
5. Feature smoke с токеном только в X-Methodist-Token и без вывода записей:
   два разных месяца → согласованные totals cases/drugs-labs; фильтр даёт ожидаемый
   пустой срез; invalid period → 422; даты ответа соответствуют выбранному периоду.
6. В отдельном процессе нового image synthetic in-memory SQLite →
   evaluate_lab_for_case: числовое значение до даты визита доступно; будущая/чужая
   запись, неизвестная единица/возраст не дают необоснованного finding; primary
   остаётся выключен. Не писать synthetic данные в production warehouse.
7. После UI release — unknown/0/partial, честные подписи, keyboard, mobile,
   сохранение периода при переходах. Снимки реальных пациентов не класть в PR/docs.
8. Если feature smoke провалился — релиз НЕ готов; штатный rollback предыдущего
   SHA и разбор причины. Health=200 сам по себе недостаточен.
9. Handoff с точными merge/production SHA, BUILD_VERSION, тестами, откатом,
   остатком и следующей безопасной командой. Не писать «весь план выполнен».

## 8. Остаток исходного плана — не потерять после первых исправлений

| Блок | Что ещё реализовать | Критерий завершения |
|---|---|---|
| Единый срез A01/A12/A28 | CohortSpec/hash, все фильтры и endpoints, №55 без подмены выборкой 120, export/drill parity, защита от устаревших ответов | Один набор случаев UI/API/export; задержанный ответ не перерисовывает новый срез |
| Анализы A02/A04/A16/A22 | available_at и тип события, локальные refs/единицы/популяция, order/result/interpretation, статусы каждого check | Нет будущего знания и ложной нормы; тесты пропусков/ошибок/детей/беременности/единиц; clinical review |
| Оценки A05/A06/A09/A13/A23 | Полный coverage/status evaluators, стабильный finding identity, SQL primary/shadow/review provenance, миграция старых projections | 100 только при выполненных применимых проверках; один факт не штрафуется дважды |
| Группы A10/A11/A24 | n группы и оценимых, явные знаменатели, small-n guard, case-mix, калиброванная уверенность | Не ранжировать n=1 как надёжное сравнение; объяснимый процент и uncertainty |
| №55/диагноз A07/A08/A15 | Утверждённый mapping критериев/ролей, evidence отдельно от заполнения, связанное действие при red flag, обоснованная коррекция/сохранение плана | Вердикт с нормой, применимостью и доказательствами, согласован с экспертами |
| История A20/A21 | Время внутри дня, episode boundaries/relevance, cross-specialty context, active medication timeline | Реально учитывается история ДО визита, без кредита за посторонний/будущий эпизод |
| Лекарства A03/A14/A17/A18/A19 | Полный assertion/subject/time, форма/путь/редакция на дату, доза конкретного назначения, indication graph, medication reconciliation | Чужая доза не закрывает пропуск; off-label не равен автоматически дефекту; review и качество по gold |
| Интерфейс A25/A26/A27/A29/A32 | Завершить inventory экранов/графиков из раздела 11 аудита, accessibility/tooltip escaping/roles, missing/error/loading, scope услуг | Полные сценарии desktop/mobile/keyboard, понятные n/N/происхождение, отдельный МО E2E |
| Архитектура | Fact model, immutable EvaluationRun, lineage источников/версий/flags, разделение extraction/applicability/evaluation/aggregation | Любой score воспроизводим; ошибка модуля видна и не выглядит успешной проверкой |
| Данные и обучение A30 | Eligibility/consent policy отдельно от CRM verdict, отзыв, patient/time split, неизменяемый holdout | Нет автоматического включения всех разборов в обучение и утечки пациента в test |
| Эксплуатация A31 | Довести #214 с #216, artifact checks, мониторинг срезов/coverage/ошибок, rollback, measured performance | Один канон, безопасные ветки, проверенный образ, наблюдаемая деградация |

Раскрытие каждого блока, источники и соответствующие файлы — в полном аудите.
История визитов обязательна по прямому требованию пользователя: не сокращать её
до одного флага «повторный визит». Не считать частичное исправление prior в #208
завершением всей продольной истории.

Дальнейшие этапы после технического batch: evidence contracts → история и
назначения → интерфейс → blind validation/promotion → масштабирование.
Новые клинические веса/primary требуют решений из clinical-review-gates.md.
Клинический владелец пользователем ещё не назван; отсутствие ответа не approval.
Продолжать независимую техническую работу, но не симулировать врачебную валидацию.

## 9. Экономия бюджета и контроль выполнения

Не повторять исходный аудит и уже зелёные неизменившиеся тесты без причины.
Сильную модель использовать для клинических контрактов, архитектуры и review;
обычную — для ограниченных механических изменений. Перед сменой модели
сохранить PR/SHA/handoff. Модель не заменяет тесты и экспертов.

Для каждого этапа вести таблицу: требование → файлы/PR → тест → результат →
merge SHA → production SHA → остаток. Статусы: planned / implemented /
locally_verified / ci_verified / merged / deployed / clinically_validated.
Не перескакивать от implemented сразу к clinically_validated.

Пользователь просит продолжить подробно и проверять после каждого этапа.
Это не разрешение обходить CI, включать все primary flags, менять нормативные
веса без review или запускать дорогой backfill с Mac. Gemini/night LLM — GCE.

## 10. Первая безопасная команда и критерий финального ответа

```bash
cd /Users/pavelkuzauka/Cursor_Folders/Protocol
git status --short --branch
```

Дальше read-only preflight и чтение перечисленных документов. На main не править.
В первом ответе сообщить пользователю актуальные main/prod SHA, состояние
packaging blocker, порядок принятых PR и место журнала. Затем выполнять работу.
Финальный результат разделить: что реально deployed, что только merged/PR,
какие clinical gates ещё не пройдены. Обещать абсолютное отсутствие сбоев нельзя;
обеспечить проверяемый процесс, остановку при расхождении и безопасный откат.
