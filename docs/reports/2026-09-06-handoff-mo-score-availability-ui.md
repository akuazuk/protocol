# Handoff: МО — видимость неполных оценок

2026-09-06; akuazuk/protocol; agent1 / pc1.
Branch codex/mo-score-availability-ui-agent1-pc1.
Worktree /Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/mo-score-availability-ui (locked).
Base a592d588fdd7eb428161024ad13e4e3948bb3754; HEAD — commit с этим handoff, SHA в PR.

## Изменение

Карточки лекарств/анализов сохраняются при пустом payload или двух null.
Вместо исчезновения — «Не оценено», «Недостаточно данных для оценки».
Реальный ноль отображается как 0 / 100. Статус partial/completed отличим от
старого payload, где полнота проверки не подтверждена. Note режима сохранён.

## Проверки и ограничения

35 passed: test_mo_frontend_structure, test_mo_ui_phase2, test_frontend_escaping_guard.
Node syntax и git diff --check пройдены. Synthetic browser проверил 4 состояния
через настоящий обработчик открытия карточки и API fixture; page errors=[];
mobile row clientWidth=scrollWidth=318 при viewport 390. Снимок просмотрен.
Harness и изображение в assets/mo-score-availability-ui/. Это не production smoke.
Полный UI/UX roadmap не закрыт этим изменением; status полноты зависит от #207.

## Координация

BUILD_VERSION 2026-09-06-090856Z-mo-score-availability-ui.
Merge/deploy нет. Последний проверенный production a592d588; #205 уже merged
fe0734a8, но не deployed нами. Данные и primary flags не менялись.
Зона владения frontend/web/shared/mo-app.js, этот handoff и assets.
В rag_server.py только BUILD_VERSION. После #207 синхронизировать с main,
сохранив API #205 и прочие изменения. До снятия draft не merge.
Не удалять активный locked worktree. Полный аудит/tracker в #209.

Следующая безопасная команда:

```bash
gh pr list --repo akuazuk/protocol --state open
```

## Дополнение: происхождение уверенности

Убраны выдуманные 90/75/55 процентов уверенности по parse_ok/date_mismatch/осям.
Убраны 100% полноты только по parse_ok. Fallback наличия осей подписан
«Заполненность осей», не полнота клинической проверки. Уверенность API сохранена
с пояснением, что она не подтверждает точность медицинского вывода.
Повторно 35 passed и 4 browser payload (в том числе parse_ok=1 без confidence);
BUILD_VERSION 2026-09-06-091452Z-mo-score-availability-ui. Merge/deploy ещё нет.

## Дополнение: таблицы семейств

Проценты таблиц подписаны как доля всех МО периода. Убрано неподтверждённое
обещание ≥20; распределение замечаний не представлено рейтингом врачей.
В таблице кодов native button даёт keyboard drill через существующий handler.
41 passed с добавлением test_mo_meds_labs_dashboards; browser Enter открывает
page=documents с finding_codes=B_lab_unused_in_dx. Group-specific denominator,
малые выборки и сравнение качества ещё требуют backend контракта; здесь
исправлена подпись текущего фактического расчёта. Добавлен файл под владение:
frontend/web/methodist/mis-kz-quality.html (проверен свободным до изменения).
