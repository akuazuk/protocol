# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-21  
Репозиторий: `akuazuk/protocol`  
Канон: `origin/main` `bf1e26e` (#167)  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Worktree: `/private/tmp/protocol-task-mo-kp-dx-only-pc1`  
ветка `cursor/mo-kp-dx-only-pc1`.

Primary: `https://protocol.kravira.by`  
Прод: **`2026-08-20-061113Z-mo-grade-ui`**.

---

## Сделано

- `kp-eval-full` exit 0, отчёт `kp_suggest_eval_post165.json` (13:20 UTC 20.08).
  Hit 67.4%, омнибус 18, ЛОР 2017 исчез. Чужие top-1: ПЦД ×90, ГСК ×53, экстренка детям ×83.
- Оценки пяти уровней в UI. Подбор КП **не** «диагноз → рейтинг протоколов».
- План `docs/plans/2026-08-21-mo-kp-diagnosis-only-v1.md`. В этой ветке: нет поиска по жалобам; индекс кандидатов без тела PDF.

## Делается

PR ветки: жалобы выключены, индекс без PDF. Нужен merge + one-off sample 300, не рестарт ради eval.

## Нужно

1. Merge PR, sample 300 на GCE (как `kp-eval`, не `protocol-web`).
2. Если ПЦД/ГСК/экстренка = 0 - полный CSV и deploy.
3. Не включать `MO_RCETH_LABEL_PRIMARY`.

## Запрет

- Второй full Rceth parse, Gemini с Mac, push в `main`, PHI
