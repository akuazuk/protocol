# Пилот 10 КЗ (2026-06-18)

**Источник:** `~/Downloads/КЗ` → `clients_consult/report_*.pdf`  
**Скрипт:** `python3 scripts/run_kz_pilot_batch.py --import-from ~/Downloads/КЗ`

## L1 (локально, 10/10 OK)

| case_id | overall % | rules % | failed | приоритет |
|---------|-----------|---------|--------|-----------|
| report_2 | 59.3 | — | 0 | **низкий балл** |
| report_4 | 61.9 | — | 0 | **низкий балл** |
| report_6 | 82.4 | 0 | 2 | **failed rules** |
| report_3 | 86.9 | 50 | 1 | правило |
| report_1 | 88.1 | — | 0 | |
| report_5 | 87.4 | — | 0 | |
| report_9 | 90.6 | — | 0 | |
| report_10 | 90.4 | — | 0 | |
| report_7 | 92.6 | 100 | 0 | |
| report_8 | 95.9 | — | 0 | |

## Этапы

1. ✅ L1 batch + snapshots (`analysis_id` в `report.json`)
2. ⚠️ AI-review локально — geo block Gemini; на Render работает
3. 🔄 `submit_priority_reviews_render.py --from-report` → feedback на Render

## Разметка методиста

- UI: **Очередь** → **Открыть** по `analysis_id` из `PILOT_REPORT.md`
- Или дождаться `render_reviews.json` и проверить rating/verdict

## Повтор

```bash
python3 scripts/run_kz_pilot_batch.py --compare-tiers   # L0/L1/L2
python3 scripts/run_kz_pilot_batch.py --submit-render     # L1+AI+approve на Render
python3 scripts/run_kz_pilot_batch.py --export-feedback   # datasets для движка
```
