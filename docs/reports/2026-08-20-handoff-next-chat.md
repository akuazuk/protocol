# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-20  
Репозиторий: `akuazuk/protocol`  
Канон: `origin/main` `e8dcfea` (#164)  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Worktree: `/private/tmp/protocol-task-kp-omnibus-content-leak-pc1`  
ветка `cursor/kp-omnibus-content-leak-pc1`.

Primary: `https://protocol.kravira.by`  
Прод runtime всё ещё **`2026-08-14-123546Z-kp-golden-40`**. В `main` уже #159-#164.

---

## Сделано

- Полный CSV на коде `e8dcfea` без рестарта UI: hit **68.9%** (5243/7605), omnibus **264 (5.0%)**, adult child-only **0**. Отчёт `kp_suggest_eval_post164.json` (31685 с).
- Было 14.08: 71.2% / 568 омнибус / 6 ложных дет_нас. Hit упал за счёт честного отказа от ЛОР 2017 (426→170) и урологии 2011.
- Новые нозологические top-1: синусит 2025 ×131, риносинусит 2024 ×63.

## Делается

В этой ветке: омнибус не читает content-index (ЛОР 2017 удерживался overlap по телу PDF).

## Нужно

1. Merge этого PR, затем one-off eval (не `protocol-web`).
2. Координатор: `bash deploy/gcp-app/deploy_to_gce.sh`. Пока UI на старом коде, методист видит матчер 14.08.
3. Калибровка Rceth 30 кейсов после deploy. Не `MO_RCETH_LABEL_PRIMARY`.
4. Ежедневно: `bash scripts/ops/daily_mo_quality_check.sh`.

## Запрет

- Второй full Rceth parse / `RCETH_PARSE_FORCE=1`
- Gemini с Mac, push в `main`, грязный checkout, PHI
