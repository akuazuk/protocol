# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-21  
Репозиторий: `akuazuk/protocol`  
Канон: `origin/main` `5cb75f9` (#175)  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Worktree: `/private/tmp/protocol-task-mo-kp-icd-mentions-pc1`  
ветка `cursor/mo-kp-icd-mentions-pc1`.

Primary: `https://protocol.kravira.by`  
Прод: **`2026-08-21-062249Z-kp-omnibus-norm`**.

---

## Сделано

- `#169`-`#175` в main. КП от диагноза/МКБ. Sample 300 после #174: hit 40.3%,
  чужие top-1 из цели = 0.
- Волна 5 в этой ветке: `icd10_mentions`, Y/W/V/T не в suggest/индексе.

## Делается

`kp-eval-full` на GCE (старт ~06:35 UTC, ~8 ч) →
`/var/data/medical_exams/reports/kp_suggest_eval_post174.json`.

## Нужно

1. Дождаться full eval. Сверить ПЦД/ГСК/экстренка/нейрохирургия/миелома = 0.
2. Merge PR mentions, sample 300, деплой **после** exit 0 у `kp-eval-full`.
3. Не включать `MO_RCETH_LABEL_PRIMARY`.

## Запрет

- Рестарт `protocol-web` пока `kp-eval-full` running
- Второй full Rceth parse, Gemini с Mac, push в `main`, PHI
