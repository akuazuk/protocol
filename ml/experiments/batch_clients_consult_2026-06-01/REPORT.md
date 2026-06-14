# Batch clients_consult — triage (2026-06-01)

**Tier:** L1 · **AI-review:** off (см. `priority_ai_reviews.json` после `review_batch_priority_cases.py`)  
**Папка:** `clients_consult/` · **13/13 OK**

## Сводка

| overall % | Кейсов |
|-----------|--------|
| ≥85% | 9 |
| 60–62% (sparse neuro) | 2 |
| 80–84% | 2 |

## P1.7 Sparse KZ — **отложено**

| case_id | overall % | failed rules | Решение |
|---------|-----------|--------------|---------|
| report_n_2 | 61.8 | 0 | Caps r109 OK; P1.7 не нужен без tag `score_misleading` |
| report_n_1 | 60.0 | 0 | `manual_review_required` — разметить методистом |

## Приоритет разметки (5 кейсов)

| case_id | analysis_id | overall % | Заметка |
|---------|-------------|-----------|---------|
| report_n_1 | `4ebe8ea9-c749-4f11-a238-45ff14df8e63` | 60.0 | sparse neuro, проверить caps |
| report_n_2 | `86493b23-37ce-427d-b0e3-3425bea12870` | 61.8 | эталон sparse после r109 |
| kard_1 | `b64da034-4a13-40af-8f93-a070601ee25c` | 87.4 | E55.9 + эутиреоз; thyroid formula FP → r116 gate |
| gastro_1 | `c826c73c-fbd5-4e3d-90d5-98214c440b93` | 89.8 | dyspepsia formula |
| F_1_p | `552f8213-26f5-41bb-a59d-b830d8eae9c9` | 85.8 | DVT formula (ожидаемо) |

**UI:** кабинет методиста → вкладка **Очередь** → **Открыть** (снимки в `data/ml/analyses/`).

**AI-предразметка:** локально Gemini недоступен (`User location is not supported`) — запустите на Render:
`python3 scripts/review_batch_priority_cases.py` или AI-review в UI после **Открыть**.

## Engine fixes (следующий спринт)

- ~~kard_1: thyroid на E55 primary~~ → **r116**
- gastro_1: `functional_dyspepsia` formula — проверить после разметки
- pl_1_d: dermatitis formula — контекст

## Команды

```bash
# AI-предразметка (нужен GOOGLE_API_KEY + METHODIST_AI_REVIEW=1)
python3 scripts/review_batch_priority_cases.py

# Pull feedback с Render
export METHODIST_TOKEN='…'
./scripts/pull_methodist_feedback.sh https://protocol-bimy.onrender.com

# Golden pairs
python3 scripts/build_golden_protocol_pairs.py --feedback-dir data/ml/feedback_render
```

## Чистые baseline

`mg_1`, `pediatr_1`, `report_g_1` — 0 failed rules.
