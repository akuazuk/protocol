# Handoff: аудит МО Аналитики, доступ Gemini с GCE, план подбора КП

Дата: 2026-09-25
Репозиторий: `origin/main` не менялся этой сессией. Коммитов и PR нет. Deploy нет.
Worktree: `/Users/pavelkuzauka/Cursor_Folders/Protocol` (грязный: новые файлы планов).
Прод: GCE `protocol.kravira.by`, контейнер `protocol-web`; версия не менялась.

## Сделано

- Проверен доступ к Gemini из контейнера `protocol-web` на GCE: список моделей
  открывается по любому ключу; `generateContent` и `embedContent` отвечают только через
  `GENERATIVE_LANGUAGE_API_KEY`. `GOOGLE_API_KEY` и `GOOGLE_API_KEY_2` - `HTTP 429`,
  месячный spend cap. `gemini-embedding-2` доступна (3072). `gemini-2.5-flash` на
  оплачиваемом ключе - `404`, модель закрыта для новых пользователей.
- План подбора КП: `docs/plans/2026-09-25-mo-kp-embed-passport-v1.md` (паспорт из Ilex
  сначала, эмбеддинг только при пустой лексике; 1875 карт реестра имеют один и тот же
  обрезанный title).
- Аудит МО Аналитики с цифрами: `docs/plans/2026-09-25-mo-analytics-audit-and-redesign-v1.md`.
  Тайминги 40 вызовов API с GCE, профиль склада за сентябрь (10 103 clinical),
  состав findings, использование CRM, инвентарь всех экранов / фильтров / графиков /
  таблиц / блоков разбора с вердиктом, протокол проверки (§5), план изменений (§6).
- `docs/plans/README.md`: две новые строки; план 2026-09-19 помечен «W0-W2 active,
  W3-W7 superseded».

Ключевые цифры для следующего агента: `/overview?month` 14.8 с; `/cases?queue_only=1`
5.0 с; `/cases/{id}` 0.15 с; `/protocol-suggest` 3.6 с холодный; `overall_grade`
fair 58 % / poor 37 % / good 3.7 % / critical 0; `zone1=weak` 71.5 %; `zone2b=na` 62.6 %;
`B_patient_history_context` P2 у 75 % случаев; CRM за сентябрь 0 строк; решений
методиста 9 за всё время.

## Не сделано

- Код не менялся. Тест `tests/test_mo_cases_query_contract.py` не создан.
- Слепая разметка прецизионности findings (§5.4 плана) не начата.
- Скриншоты прода под учёткой методиста не снимались.
- Планы не в git, пока владелец не попросит коммит.

## Нельзя параллельно

`frontend/web/shared/mo-app.js`, `frontend/web/methodist/mis-kz-quality.html`,
`rag_server.py` `api_methodist_mo_cases`, `clinical_knowledge/mo_backend.py`
`_warehouse_records` / `_filter_records` / `build_overview`, `clinical_knowledge/mo_overall_grade.py`.

## Следующая команда

```bash
scripts/ops/git_task_start.sh mo-find-cases-w0 --pc=1 \
  --branch=cursor/mo-find-cases-w0-pc1
```

Только W0 плана 2026-09-19 (правда фильтров). Затем волна A нового плана
(инструменты аудита, отдельный PR уровня 4). Не деплоить и не рестартовать
`protocol-web` во время Rceth `running`.
