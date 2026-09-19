# Handoff: план UX МО Аналитики

Дата: 2026-09-19
Репозиторий: `origin/main` не менялся этой сессией (план не коммитился).
Прод: GCE `protocol.kravira.by`, SHA предыдущего релиза ilex; UI аудировали под admin.

## Сделано

- Повторный проход всех экранов и оставшихся кнопок на проде (список, очередь, пресеты, пагинация, колонки, врачи, справка, команды).
- Корневые причины: `overall_grade` нет в FastAPI Query; «Только критические» ставит `statuses=critical` (CRM); `attach_overall_grade` копирует dict; список 7–16 с из полного scan 7809; `request()` без legacy → `mis-kz-qualityundefined`.
- План: `docs/plans/2026-09-19-mo-analytics-ux-rebuild-v1.md` (волны W0–W7).
- Индекс: строка в `docs/plans/README.md`.
- Канвас: рядом с чатом, файл `mo-analytics-ui-redesign.canvas.tsx`.

## Не сделано

- Код не менялся, PR нет, деплоя нет.
- План не в git, пока владелец не попросит коммит.

## Нельзя параллельно

`frontend/web/shared/mo-app.js`, `frontend/web/methodist/mis-kz-quality.html`, `rag_server.py` `api_methodist_mo_cases`, `clinical_knowledge/mo_overall_grade.py`, `clinical_knowledge/mo_backend.py` `_warehouse_records` / `_filter_records`.

## Следующая команда

```bash
scripts/ops/git_task_start.sh mo-find-cases-w0 --pc=1 \
  --branch=cursor/mo-find-cases-w0-pc1
```

Только W0: правда фильтров. Не начинать редизайн меню до merge W0.
