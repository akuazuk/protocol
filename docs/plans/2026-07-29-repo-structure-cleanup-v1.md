# План: cleanup структуры репозитория (frontend/backend/ops) без регрессий

**Дата:** 2026-07-29  
**Статус:** active  
**Цель:** привести структуру проекта к более чистой и предсказуемой (в стиле инженерных
команд уровня Big Tech), не ломая production-пути, Render deploy и текущий workflow двух ПК.

---

## Контекст

Сейчас в корне смешаны:
- backend entrypoint (`rag_server.py`);
- frontend assets (`index.html`, `patient.html`, CSS/JS);
- batch/ops scripts;
- data/corpus artifacts.

Это усложняет навигацию, код-ревью, поддержку несколькими компьютерами и безопасный деплой.

---

## Принципы миграции

1. **Совместимость first**: ни один прод URL не ломаем.
2. **Малые шаги**: сначала канонические entrypoints и документация, затем фактический перенос.
3. **Один смысл - один каталог**: frontend / backend / scripts / deploy / docs / data / tests.
4. **Никаких “big-bang move”** в одном коммите.

---

## Фазы

### Phase 1 (сделано в этой итерации)

- [x] Добавлен канонический backend entrypoint `backend/server.py`.
- [x] Обновлён `Procfile` на `uvicorn backend.server:app`.
- [x] Добавлены `backend/README.md` и `frontend/README.md`.
- [x] Добавлен placeholder `frontend/web/`.

### Phase 2 (выполнено частично в этой итерации)

- [x] Добавить backend-совместимость путей frontend (`backend/frontend_paths.py`) с приоритетом
  `frontend/web/*` и fallback в root (без поломки legacy URL).
- [x] Перенести frontend-статику в `frontend/web/` по группам:
  - doctor/methodist/patient/shared;
  - сохранить legacy URL через явные маршруты и fallback.
- [ ] Вынести frontend route-конфиг в отдельный backend-модуль (остаток Phase 2).

### Phase 3 (выполнено)

- [x] Добавить hygiene-аудит: `scripts/check_repo_hygiene.sh`.
- [x] Ужесточить `.gitignore` для локальных heavy/generated/log артефактов.
- [x] Упорядочить scripts по доменам (этап 3.1):
  - добавлен канонический домен `scripts/ops/` для git/deploy/hygiene команд.
- [x] Оставить совместимые shim-скрипты для старых путей:
  - существующие root-path команды `scripts/*.sh` сохранены;
  - добавлены совместимые entrypoints `scripts/ops/*.sh`.
- [x] Допривести остальные домены (этап 3.2):
  - добавлены канонические entrypoints в `scripts/dev/`, `scripts/data/`, `scripts/deploy/`;
  - совместимость root-path `scripts/*.sh` сохранена.

### Phase 4

- [x] Финальный audit root (минимум файлов на верхнем уровне):
  - зафиксирован инвентарь top-level и зоны следующего переноса.
- [x] Обновить runbook, CI и smoke-checks:
  - добавлен `scripts/ops/smoke_repo_layout.sh`;
  - добавлен CI step для smoke-check wrapper-структуры;
  - runbook дополнен отдельной командой smoke-проверки.

### Phase 5

- [x] Добавить каноничные Python-entrypoints по доменам без риска регрессий:
  - `scripts/data/py/*` для data/corpus legacy top-level scripts;
  - `scripts/dev/py/*` для diagnostic/dev legacy top-level scripts;
  - запуск через `runpy` с сохранением старых путей.
- [x] Расширить smoke-check структуры:
  - `py_compile` для `scripts/data/py/*.py` и `scripts/dev/py/*.py`.

### Phase 6

- [x] Добавить явное правило для Render main-ветки:
  - если работа велась не в `main`, перед deploy продвигать `HEAD` в `origin/main`;
  - проверять совпадение прод-версии с локальным `BUILD_VERSION`.
- [x] Добавить инструмент и документацию:
  - `scripts/ops/render_promote_main.sh`;
  - обновить runbook/README и постоянное правило в `.cursor/rules/git-push.mdc`.

---

## Критерии приёмки

1. `uvicorn backend.server:app` поднимает API без изменений поведения.
2. Render deploy работает через git-connected branch как раньше.
3. Все ключевые UI URL (`/`, `/methodist/mo`, `/patient.html`) продолжают открываться.
4. Структура проекта документирована и понятна новому разработчику за 3-5 минут.

