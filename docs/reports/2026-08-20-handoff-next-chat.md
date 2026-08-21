# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-21  
Репозиторий: `akuazuk/protocol`  
Канон: `origin/main` `2a2d6e0` (#170)  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Worktree: `/private/tmp/protocol-task-mo-kp-omnibus-pc1`  
ветка `cursor/mo-kp-omnibus-pc1`.

Primary: `https://protocol.kravira.by`  
Прод: **`2026-08-20-061113Z-mo-grade-ui`** (#169/#170 не в UI).

---

## Сделано

- `#169` merged: жалобы не ищут КП; индекс без тела PDF.
- `#170` merged: `_passes_dx_gate` - primary-корень или название.
- Sample 300 (`kp_suggest_eval_sample_post170.json`): hit **61.3%**,
  омнибус 0, adult+child 0, жалобы 0, ПЦД/ГСК/экстренка **0**.
  Чужой top-1: нейрохирургия 2021 №117 ×13 (overlap по телу PDF).

## Делается

В этой ветке: overlap без PDF body; «нейрохирургического профиля» = омнибус.

## Нужно

1. Merge PR, sample 300. Если нейрохирургия = 0 - полный CSV и `deploy_to_gce.sh`.
2. Не включать `MO_RCETH_LABEL_PRIMARY`.

## Запрет

- Второй full Rceth parse, Gemini с Mac, push в `main`, PHI
