# Protocol ML: непрерывное дообучение

Каркас для локальных моделей в контуре РБ. **Детерминированное ядро** (`send_gate`, `compliance_gate`, CISZ) не заменяется нейросетью.

## Принцип

| Слой | Механизм | Обучение |
|------|----------|----------|
| L0 / gate | Правила 482+ | Только human-in-the-loop в каталоге правил |
| RAG | Embed + rerank | Fine-tune / LoRA на парах запрос-чанк |
| Термины КЗ | Entailment matcher | Метки методиста (required_exam) |
| Summary cards | Structured extractor | Approved drafts → student model |
| L2 текст | Explainer | JSON compliance → текст (optional) |

## Структура

```
ml/
  README.md                 # этот файл
  configs/default.json      # модели, пути, пороги eval
  registry/model_manifest.json
  datasets/                 # экспорт из scripts/export_training_feedback.py
  train/                    # скрипты fine-tune (заглушки CLI, --dry-run)
  eval/                     # регрессия vs golden sets
data/ml/feedback/           # сырые события пилота (JSONL, без ПДн)
```

## Сбор feedback

События пишутся в `data/ml/feedback/*.jsonl` (одна строка = один JSON):

- `l0_screen` - hash текста, scores, latency, rubric
- `retrieval_fix` - query, rejected_path, chosen_path
- `methodist_override` - rule_id, expected pass/fail
- `summary_edit` - protocol_id, review_status

Экспорт датасетов:

```bash
python3 scripts/export_training_feedback.py
python3 scripts/export_training_feedback.py --seed-only   # только bootstrap из golden
```

## Статус (2026)

| Компонент | Статус |
|-----------|--------|
| send_gate, правила 482+ | Production |
| Semantic RAG, FAISS, hybrid retrieve | Production |
| Опциональный LLM API (L2, enrichment) | Production (при наличии ключа) |
| `data/ml/feedback/`, `export_training_feedback.py` | Каркас, seed-датасеты |
| `ml/train/*`, `ml/eval/*` | Заглушки CLI (`--dry-run`) |
| LoRA fine-tune + деплой local embedder | Roadmap фаза 1 (Q4 2026) |

## Цикл MLOps (целевой)

1. Production L0-L2 → `data/ml/feedback/`
2. `export_training_feedback.py` → `ml/datasets/*.jsonl`
3. `ml/train/finetune_embedder.py` (LoRA) → checkpoint
4. `ml/eval/run_regression.py` vs `data/quality_benchmark.json` + `consult_gold.jsonl`
5. Pass → запись в `ml/registry/model_manifest.json`, деплой sidecar

## Env (будущее)

- `RAG_EMBED_BACKEND=local|cloud`
- `ML_MODEL_REGISTRY=ml/registry/model_manifest.json`
- `ML_FEEDBACK_DIR=data/ml/feedback`

См. также `docs/architecture-stages-print.html` §11-12 (семантический RAG, `kz_quality_scores`).

## Эксперимент embedder 001

Первый прогон fine-tune на seed-датасете (313 пар):

```bash
pip install -r requirements-ml.txt
python3 scripts/run_embedder_experiment.py
```

Отчёт: `ml/experiments/embedder_exp_001/REPORT.md` (чекпоинты локально, в git не коммитятся).
