# План: топический DDI не поднимать в «Критично»

Статус: **completed** (код в `main` #138+#139; хвост данных - backfill по запросу)  
Дата: 2026-08-11  
Кейс: визит `3665385` - Ксарелто + диклофенак **гель** попал в очередь как Критично.  
Handoff: `docs/reports/2026-08-11-handoff-mo-ddi-topical-and-agent-sync.md`

## Цель

Major DDInter, где один партнёр топический (гель/мазь/местно/свечи), не даёт полосу
очереди **Критично** и в оценке идёт как **Умеренно (P2)**.

## Изменения (сделано)

- `medication_safety`: детект топического упоминания ЛС
- `kz_deep_eval`: Major + topical → P2, `topical_ddi=true`
- `mo_action_queue_select`: topical DDI вне очереди (как Moderate)
- `mo_zone_scores`: P2/P3 safety findings не поднимают attention `important` (#139)

Системный диклофенак + антикоагулянт остаётся Major / Критично в очереди.

## Метрики

| | Было | Стало | Цель |
|--|------|-------|------|
| 3665385 queue / attention | Критично / safety important | вне очереди, attention none | топический Major не в Critical |
| 3665385 overall / status | ~60 review | 87.7 good | после re-eval без stale v4 |
| 3651370 sertraline+amitriptyline | Major P1 | без изменений (верно) | системные пары не demote |

## Шаги

- [x] Код + тесты + PR #138
- [x] Zone attention P2 fix PR #139
- [x] Deploy GCE, smoke 3665385
- [ ] Backfill других дат со старым P1 topical DDI (только по OK владельца)
