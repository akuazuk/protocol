# Handoff для второго компьютера (2026-08-03)

Этот файл фиксирует, что сделано сегодня по МО BI и что нужно сделать на другой машине,
чтобы подхватить контекст и продолжить без потерь.

## 1) Что уже в origin/main

Коммиты за сегодня, уже в `origin/main`:

- `3ba03bc` - token-aware открытие PDF + batch PDF + column manager
- `0f4c299` - устранение popup-block, санитизация hash-диагнозов, fallback case detail

Текущая прод-версия после последнего деплоя: `2026-08-03-r10-mo-bi-pdf-popup-diagnosis-fix`.

## 2) Что добавлено локально в этой итерации (ожидает push)

### Исправления case PDF / документа МО

- `clinical_knowledge/mo_case_document.py`
  - приоритет `medical_exam` при выборе типа документа;
  - нормализация id (`3621757.0` -> `3621757`) при поиске строки в parquet;
  - поиск source-строки не только по входному case_id, но и по `visit_id`/`mis_id` из витрины;
  - санитизация МКБ: hash-like значения не попадают в PDF;
  - выбор первого валидного ICD-кода из meta/record/source.

- `clinical_knowledge/mo_backend.py`
  - в fallback case detail приоритет выбора записи: `medical_exam` -> `consultation` -> прочее.

- `tests/test_mo_phase78_reports.py`
  - новый тест на приоритет `medical_exam`, sanitzation hash МКБ и наличие клинического текста.

## 3) Как подхватить на другой машине

```bash
cd ~/CURSOR/Protocol/protocol

# 0. Убедиться, что нет несохранённых локальных изменений
# (или сделать stash/commit)
git status

# 1. Забрать последние изменения
git fetch origin

# 2. Переключиться на main и синхронизироваться
git checkout main
git pull --ff-only origin main

# 3. Проверить версию сборки в коде
rg -n "BUILD_VERSION =" rag_server.py

# 4. Прогнать ключевые тесты МО
python3 -m pytest tests/test_mo_phase78_reports.py tests/test_mo_backend.py -q
```

## 4) Что делать дальше (следующий шаг разработки)

1. Проверить проблемный кейс в проде:
   - `/api/methodist/mo/cases/3621757/pdf`
   - убедиться, что заголовок документа = «Медицинский осмотр», МКБ не hash,
     блок «Текст МО» заполнен клиническими полями.
2. Если на проде нет клинического текста:
   - проверить наличие `raw/YYYY/MM/mo_YYYY-MM-DD.parquet` для даты визита;
   - проверить, что `visit_id`/`mis_id` реально присутствует в parquet;
   - при необходимости расширить fallback на secure CSV и/или отдельный API МИС текста.
3. Добить heavy BI backlog (если не закрыт полностью):
   - compare-mode по нескольким периодам;
   - пакетная выгрузка PDF/XLSX по очереди;
   - drill-down матрицы и сценарии weekly/monthly экспортов.

## 5) Важные примечания

- Папка `data/ml/reports/v4/` в корне может быть untracked локально - не коммитить в feature/UI фиксы.
- Для release-коммитов обязательно поднимать `BUILD_VERSION` в `rag_server.py`.
- После коммита - всегда `git push origin`, затем проверка деплоя:

```bash
scripts/ops/render_promote_main.sh --prod-url=https://protocol-bimy.onrender.com
```
