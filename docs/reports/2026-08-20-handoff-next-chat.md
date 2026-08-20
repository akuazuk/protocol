# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-20  
Репозиторий: `akuazuk/protocol`  
Канон: `origin/main` `419cbbf` (#165)  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Worktree: `/private/tmp/protocol-task-mo-grade-ladder-pc1`  
ветка `cursor/mo-grade-ladder-pc1`.

Primary: `https://protocol.kravira.by`  
Прод runtime всё ещё **`2026-08-14-123546Z-kp-golden-40`**.

---

## Сделано

- План пяти уровней: `docs/plans/2026-08-20-mo-grade-ladder-v1.md`.
- Прогон склада 9736 clinical (26.07-19.08): zone1 слабо 74%, КП matched 0,
  unmatched+bad план 3946, внимание safety 216.
- Код `compute_mo_overall_grade` + тесты; поле в JSON зон.

## Делается

`kp-eval-full` на GCE после #165 (отчёт `kp_suggest_eval_post165.json`).

## Нужно

1. Merge PR этой ветки.
2. Координатор: `deploy_to_gce.sh`.
3. Волна 1 UI чипа - отдельный PR. Rceth в итог не включать.
4. Калибровка Rceth 30 кейсов после deploy.

## Запрет

- Второй full Rceth parse / `RCETH_PARSE_FORCE=1`
- Gemini с Mac, push в `main`, грязный checkout, PHI
