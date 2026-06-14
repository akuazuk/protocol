# Backlog ML-плана (когда накопятся КЗ)

**Ставится на паузу до новой порции PDF в `clients_consult/`.**  
Связанный план: [methodist-ml-priority-plan.md](./methodist-ml-priority-plan.md)

## Сразу после новой партии КZ

1. `run_methodist_batch.py --folder clients_consult --tier L1`
2. Verify batch на Render (local = Render delta 0)
3. AI-review спорных кейсов → `retrieval_fix` с путями `minzdrav_protocols/...`
4. `pull_methodist_feedback.sh` → `build_golden_protocol_pairs.py`
5. Engine PR по `priority_cases` + запись в `engine_release_log.json`

## Цели данных (ещё не закрыты)

| Метрика | Сейчас (~июнь 2026) | Цель |
|---------|---------------------|------|
| `retrieval_fix` | 9 | ≥20 (3 мес), ≥50 (LoRA) |
| golden_protocol_pairs | 14 | ≥20 |
| unique KZ | ~28 | ≥50 |
| readiness | ~52% | >60% |

## Engine backlog (r122+)

- `report_n_1` / `report_n_2` — sparse neuro caps vs safety
- `pl_*` — superficial vein I80.x rules
- `report_procy_g_1` — oncology routing vs peptic ulcer rules
- Нормализация `retrieval_fix` (только catalog paths, не абстрактный текст AI)
- Gates для matched-path rules (продолжение r121)

## ML (после ≥50 retrieval_fix)

```bash
ML_FEEDBACK_DIR=data/ml/feedback_render python3 scripts/export_training_feedback.py
python3 ml/train/finetune_embedder.py --dataset ml/datasets/retrieval_pairs_resolved.jsonl
python3 scripts/run_ab_embedder_kz.py
```

## Еженедельный ритуал (30 мин)

```bash
set -a && source .env && set +a
./scripts/pull_methodist_feedback.sh https://protocol-bimy.onrender.com
.venv/bin/python scripts/build_golden_protocol_pairs.py --feedback-dir data/ml/feedback_render
.venv/bin/python scripts/analyze_priority_cases.py  # triage для Cursor
```
