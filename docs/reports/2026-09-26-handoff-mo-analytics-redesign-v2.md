# Handoff: повторный аудит МО Аналитики и план редизайна v2

Дата: 2026-09-26
Репозиторий: `akuazuk/protocol`
Ветка: `cursor/mo-analytics-redesign-v2-pc1`
Worktree: `/Users/pavelkuzauka/Cursor_Folders/Protocol-worktrees/mo-analytics-redesign-v2`
База: `origin/main` `0bb084cf`
HEAD: `0b2bd559`, PR: https://github.com/akuazuk/protocol/pull/295 (docs only)
Прод на момент аудита: `8000354f`, `2026-09-21-184253Z-warehouse-lock-alerts`
Merge / deploy в этой сессии: **нет**. Код не менялся, только docs.

## Сделано

- Повторный аудит прода в браузере под сессией методиста: все 9 экранов + разбор случая,
  1024 и 1440, DOM-аудит (карточки, ECharts, таблицы, overflow, шрифты), тайминги всех
  `/api/methodist/mo/*`, 26 проб поиска, распределения склада на GCE.
- План `docs/plans/2026-09-26-mo-analytics-redesign-v2.md`: требования владельца
  (ничего не скрывать, много стильных дашбордов, шрифты, всё работает, данные с 1 января
  и догрузка по кнопке, умный поиск), факты по каждому элементу, дизайн-система, каталог
  дашбордов по экранам, волны A-H, метрики было → цель.
- Вчерашние черновики v1 (`2026-09-25-mo-analytics-audit-and-redesign-v1.md`,
  `2026-09-25-mo-kp-embed-passport-v1.md`, handoff 09-25) перенесены в эту ветку с
  примечанием: код читался из checkout `e15ac9cf`, отставшего от `origin/main` на
  81 коммит; часть выводов уже закрыта W0-W7 / D0-U3. v1 аудита - archived.
- `docs/plans/README.md`: три строки сверху таблицы.

## Ключевые цифры (сентябрь 2026, clinical, прод)

- overall_grade: good 389 (3,7%), fair 6151, poor 3846, important 141, critical 0.
- zone1 ok 4,8%; zone2b ok 0 (weak 1946, bad 1983, na 6598); kp matched 3929 / 10527.
- Клинические МО с зонами только с 2026-07; январь-июнь 77 тыс. документов без оценки;
  `secure_cases` только с июня; лаборатория с 2025-12.
- `/cases` холодный 12,1 с, тёплый ~1 с; `sort_by=score` 4,5 с; очередь 5,7-6,1 с;
  `/freshness` 3,6-3,9 с.
- Поиск: `гипертония` 0 при `гипертензия` 157 и `I10` 168; опечатка 0; `близорукость` 0.
- Сверка КП МЗ стоит с 2026-08-26, Rceth с 2026-08-18 при живом cron; очередь ingest
  разбирается только в 02:00Z.
- Разбор случая: 21 `details`, 20 закрыты по умолчанию. Меню: 4 пункта в «Ещё».
- Шрифт `Avenir Next` без webfont; веса 750/850 синтезируются.

## Не сделано

- Ни одна волна A-H не начата. Кода нет.
- Скриншоты аудита лежали в `/tmp/mo_audit` (в git не кладём: ФИО врачей).
- Причина холодных 12 с не найдена - только гипотезы (волна B1).
- Причина остановки `night_kp_sync.sh` и Rceth не смотрелась в логах GCE (волна B4).

## Тесты

Не запускались: только документация. `git diff --check` чистый.

## Нельзя трогать параллельно

`frontend/web/shared/mo-app.js`, `mo-charts.js`, `mo-ui.css`, `mo-tokens.css`,
`clinical_knowledge/mo_backend.py` (поиск, пейджинг), `docs/plans/README.md`,
`rag_server.py` кроме `BUILD_VERSION`. Пересчёт января-июня (волна C) - только один
процесс на GCE и только координатор.

## Следующая безопасная команда

```bash
cd /Users/pavelkuzauka/Cursor_Folders/Protocol && \
scripts/ops/git_task_start.sh mo-redesign-a-quick-fixes --pc=1 \
  --branch=cursor/mo-redesign-a-quick-fixes-pc1 && \
python3 scripts/ops/pr_dashboard.py --files frontend/web/shared/mo-app.js frontend/web/shared/mo-ui.css
```

Запреты: не работать в `/Users/pavelkuzauka/Cursor_Folders/Protocol` на `main` (отстал на
81 коммит, грязный); не деплоить без merge; не запускать recompute января-июня без
координатора; Gemini/MIS только с GCE.
