# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-21  
Репозиторий: `akuazuk/protocol`  
Канон: `origin/main` `a6f7f7d` (#169)  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Worktree: `/private/tmp/protocol-task-mo-kp-dx-gate-pc1`  
ветка `cursor/mo-kp-dx-gate-pc1`.

Primary: `https://protocol.kravira.by`  
Прод: **`2026-08-20-061113Z-mo-grade-ui`** (ещё не #169).

---

## Сделано

- `kp-eval-full` exit 0, отчёт `kp_suggest_eval_post165.json`.
  Hit 67.4%, омнибус 18. Чужие top-1: ПЦД ×90, ГСК ×53, экстренка детям ×83.
- `#169` merged: жалобы не ищут КП; индекс без тела PDF.
- Волна 2-3 в этой ветке: `_passes_dx_gate` - карта только при
  пересечении `icd10_primary` или названия с диагнозом / RU title кода.
  `match_kind=clinical` без gate запрещён. Жалобы в matcher не идут.

## Делается

PR волны 2-3. После merge - one-off sample 300 на GCE, не рестарт UI.

## Нужно

1. Merge PR gate, sample 300 (`kp_suggest_eval_sample_post169.json`).
2. Если ПЦД/ГСК/экстренка = 0 - полный CSV и `deploy_to_gce.sh`.
3. Не включать `MO_RCETH_LABEL_PRIMARY`.

## Запрет

- Второй full Rceth parse, Gemini с Mac, push в `main`, PHI
