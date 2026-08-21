# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-21  
Репозиторий: `akuazuk/protocol`  
Канон: `origin/main` `eafe4dc` (#172)  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Worktree: `/private/tmp/protocol-task-mo-kp-whole-token-pc1`  
ветка `cursor/mo-kp-whole-token-pc1`.

Primary: `https://protocol.kravira.by`  
Прод: **`2026-08-20-061113Z-mo-grade-ui`** (#169-#172 не в UI).

---

## Сделано

- `#169` `#170` `#171` `#172` в `origin/main`.
- Sample 300 после #172 (`kp_suggest_eval_sample_post172.json`):
  hit 47.3%, жалобы 0, ПЦД/ГСК/экстренка 0, adult+child 0.
  Всё ещё: нейрохирургия ×11, миелома ×10.
- Причина: `word in blob` - «хирург» внутри «нейрохирургического»;
  омнибус проходил при overlap ≥ 0.75.

## Делается

В этой ветке: целые слова / основа (пищевод→пищевода);
омнибус в suggest не берём.

## Нужно

1. Merge PR whole-token, sample 300.
2. Если нейрохирургия и миелома-на-J06 = 0 - полный CSV и `deploy_to_gce.sh`.
3. Не включать `MO_RCETH_LABEL_PRIMARY`. Не деплоить, пока sample грязный.

## Запрет

- Второй full Rceth parse, Gemini с Mac, push в `main`, PHI
