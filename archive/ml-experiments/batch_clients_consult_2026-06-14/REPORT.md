# Batch clients_consult — 2026-06-14 (r118)

**Папка:** `clients_consult/` — **20 PDF** (7 новых с 2026-06-14)  
**Tier:** L1 local + Render AI-review для новых  
**BUILD:** `2026-06-01-r118-heuristic-facts-safety-follow`

## Сводка

| Метрика | Значение |
|---------|----------|
| Всего | 20/20 OK |
| Новых кейсов | 7 |
| Render reviews (новые) | 7/7 |
| Priority по AI (rating≤2) | 6 из 7 новых |

## Новые КЗ (Jun 14)

| case_id | overall % | failed rules | AI rating | Теги |
|---------|-----------|--------------|-----------|------|
| report_aler_1 | 83.0 | 0 | 2 | — |
| report_g_11 | 84.6 | pregnancy formula | 2 | false_positive_rule, score_misleading |
| report_g_12 | 83.0 | pregnancy formula | 2 | false_positive_rule, score_misleading |
| report_lor_1 | **55.0** | 0 | 2 | score_misleading |
| report_oftal_1 | 84.7 | 0 | **4** | — |
| report_proc_1 | 91.4 | 0 | 2 | score_misleading |
| report_ter_1 | 88.6 | **6** | **1** | wrong_protocol, false_positive_rule, score_misleading |

## r118 регресс (старые кейсы)

| case_id | r116/r117 | r118 batch |
|---------|-----------|------------|
| F_1_p | 85.8%, e912b455 FAIL | **89.3%**, 0 fails |
| gastro_1 | 89.8% | **75.0%** cap |
| pl_1_f | — | 84.6%, 0 fails |

## Следующие engine fixes (backlog)

1. **report_g_11/12** — gate `d4c0214b_path_pregnancy_diagnosis_formula` (не беременность)
2. **report_ter_1** — 6 чужих правил (гастрит/травма/инородное) на терапевтическом КЗ → rule family gates
3. **report_lor_1** — overall 55% при rules=0 → sparse/caps ЛOR
4. **P1.10** — `retrieval_fix` для report_ter_1 (`wrong_protocol`)

## Файлы

- `report.json`, `batch_summary.csv` — локальный L1
- `render_reviews_new.json` — Render AI + feedback (новые)
- `REVIEW_QUEUE.md` — очередь для UI
