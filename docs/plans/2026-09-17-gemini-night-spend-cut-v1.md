# Ночной Gemini: срезать расход (v1)

Дата: 2026-09-17  
Статус: active  
Связанные: `2026-08-09-mo-shadow-dx-plan-conservative-v1.md`,
`deploy/gcp-llm/README.md`

## 1. Контекст

В сентябре billed-проект `gen-lang-client-0274478609` накрутил порядка $93.
Склад `fact_llm_usage` видел только грейдер (~$41 по старым ценам). Основной
перерасход - shadow Dx/Plan по всем клиническим визитам (~450/день × 2 вызова)
и 100% эскалация очереди 80 на `gemini-3.1-pro-preview` из-за `needs_human`.

## 2. Что меняется в проде

| Было | Станет |
|--|--|
| Shadow без лимита | 30 кейсов/день |
| Action-judge без лимита, HTTP на Render | 20 кейсов, слоты из локального jsonl, без Gemini если документа нет |
| Pro на каждый `needs_human` | Pro только parse / low confidence / harm disagreement |
| Thinking по умолчанию | `thinking_budget=0`, JSON max 2048 |
| Spend-cap ретраился после смены ключа | терминальная ошибка |
| Flash в складе $1.50/$7.50 | intro $0.75/$3.75 до конца 2026 |
| Нет budget alert | скрипт `ensure_gemini_spend_budget.sh` на $25/$40 |

## 3. Метрики

| Метрика | Было (сент.) | Цель |
|--|--|--|
| Shadow вызовов/день | ~900 | ≤ 60 |
| Pro вызовов/день | 80 | единицы |
| Ночной Gemini USD | ~$5-7/день | ~$0.3-0.8/день |

## 4. Шаги

- [x] Лимиты runner + night pipeline
- [x] Эскалация без `needs_human`
- [x] thinking_budget=0 + Batch best-effort + context cache
- [x] Учёт thinking-токенов и intro-цены
- [ ] Merge + deploy GCE
- [ ] `bash deploy/gcp-llm/ensure_gemini_spend_budget.sh` координатором
- [ ] Smoke: одна ночь, shadow lines ≤ 30, judge без `document: HTTP`

## 5. Риски

| Риск | Митигация |
|--|--|
| Batch API 404 | sequential fallback |
| Context cache недоступен на flash | system_instruction без cache |
| Shadow 30 мало для калибровки | `MO_SHADOW_DX_PLAN_LIMIT=0` явно |
