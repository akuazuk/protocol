# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-20  
Репозиторий: `akuazuk/protocol`  
Канон: `origin/main` `3bf14f3` (#166)  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Worktree волн 1-2: `/private/tmp/protocol-task-mo-grade-ui-pc1`  
ветка `cursor/mo-grade-ui-pc1`.

Primary: `https://protocol.kravira.by`  
Прод runtime: **`2026-08-20-055730Z-mo-grade-verify`** (merge #166).

---

## Сделано

- Волна 0 в проде: JSON зон содержит `overall_grade`. Smoke HTTPS + контейнер: пустые жалобы → «Важно».
- Rceth `done` (3017/3017). Deploy `protocol-web` не убил `kp-eval-full`.
- Волны 1-2 в этой ветке: чип/фильтр/колонка; unmatched план → `na`; колонки склада + `scripts/ops/backfill_mo_overall_grade.py`.

## Делается

`kp-eval-full` на GCE (`kp_suggest_eval_post165.json`, ещё нет файла).

## Нужно

1. Merge PR волны 1-2, затем `deploy_to_gce.sh`.
2. На GCE: `docker exec protocol-web python3 /app/scripts/ops/backfill_mo_overall_grade.py`
3. Приёмка: leftover unmatched+bad = 0; чип на карточке = `label_ru`.
4. Rceth в итог не включать. Калибровка 30 - после UI.

## Запрет

- Второй full Rceth parse / `RCETH_PARSE_FORCE=1`
- Gemini с Mac, push в `main`, грязный checkout, PHI
- `MO_RCETH_LABEL_PRIMARY=1`
