# Protocol: план действий (master)

> **Обновлено:** 2026-06-29  
> **Prod BUILD:** `2026-06-29-r2-post-wave-a-checklist`  
> **Аудитория:** Павел + Cursor Agent  
> **Связанные документы:** [cursor-spend-checklist.md](./cursor-spend-checklist.md) (ROI Cursor $70), [chunk-quality-pipeline.md](./chunk-quality-pipeline.md), [ml/README.md](../ml/README.md)

---

## 0. Где мы сейчас (одна страница)

| Область | Статус | Комментарий |
|---------|--------|-------------|
| Wave A chunk QA | ✅ | 10 000 fixes, 57 852 chunks на prod |
| B1 routing (age, rubric) | ✅ на prod r2 | L1 batch **не** использует chunk-RAG |
| B2 B2C (tier, idempotency) | ✅ | 3 P0 из architecture §17 |
| L1 batch (30 KZ) | ✅ ~84.2% | без изменений после Wave A - **норма** |
| L2 smoke (5 рубрик) | ✅ ~85.6% | prod r2 |
| Search probe (100) | ✅ top-1 100% | |
| **D2 embeddings** | 🔄 ~10% | Kravira billed key, ~98k чанков |
| Слабые KZ `report_n_*` | ⚠️ ~60% | routing/scoring, не чанки |
| ML classifier/reranker | ☐ | датасет есть, train - следующий этап |

**Главный урок:** chunk QA + embed улучшают **L2, search, B2C crosscheck**, не **L1 overall** (`rag_used: false` на L1).

---

## 1. Что делает продукт: L0 / L1 / L2

### L1 - быстрая structured-проверка (~2 с)

**Использует:** парсер КЗ, **protocol_cards** (карточки, не semantic search по 57k), **482+ правил**, optional alignment по чанкам **matched PDF paths only**.

**Не использует:** `retrieve()` по корпусу, embed-rerank, Gemini для score.

**`rag_used: false`** - поэтому Wave A не поднимает L1 batch.

### L2 - контекст протокола + (опц.) LLM

| Режим | Где | RAG | LLM |
|-------|-----|-----|-----|
| **L2 lite** | prod Render | path-чанки, без full retrieve | 1× synthesize |
| **L2 full** | локально | `retrieve()` + embed rerank | несколько вызовов |

**Выигрыш от embed D2:** L2 full, search, B2C crosscheck.

---

## 2. Три кошелька (не путать)

```
GOOGLE_API_KEY / Kravira  →  chunk QA, embed, Gemini на Render
$0 (CPU, скрипты)         →  audit, merge, pytest, probe
Cursor ~$70               →  Opus-архитектура + Auto-код (B1, ML export)
```

**Cursor $70 нельзя** тратить на embed или fine-tune - только на код и дизайн.

**Kravira key** (`GENERATIVE_LANGUAGE_API_KEY`, формат `AQ.…`): billed prepay, приоритет embed. Fallback: `GOOGLE_API_KEY`, `GOOGLE_API_KEY_2`.

---

## 3. Фаза A - дожать D2 (сейчас, $0 + Kravira)

### A1. Embeddings (идёт)

```bash
# статус
python3 -c "import json; d=json.load(open('output/embed_build_state.json')); print(len(d['done_chunk_ids']))"
pgrep -fl build_chunk_embeddings

# если упал - перезапуск
pkill -f build_chunk_embeddings.py
nohup .venv/bin/python scripts/build_chunk_embeddings.py >> data/ml/reports/embed_checklist_run.log 2>&1 &
```

**Gate после 100%:**

```bash
.venv/bin/python scripts/run_symptom_icd_probe.py --local --no-gemini \
  --out data/ml/reports/symptom_icd_probe_post_embed.jsonl
# top-1 не хуже baseline (100%)
```

### A2. Upload corpus на Render (после embed)

```bash
bash scripts/upload_rich_chunks_render.sh <SSH_TARGET> --gzip
# проверка: ~57k строк rich_chunks.jsonl на диске
```

### A3. Smoke + приёмка

```bash
.venv/bin/python scripts/run_clients_consult_render_batch.py \
  --tier L2 --cases gastro_1,kard_1,pediatr_1,report_lor_1,report_urolog_1 \
  --out ml/experiments/batch_post_embed_acceptance/l2_sample
```

**Критерии:** overall ≥85%, `rag_chunks_n > 0` где возможно, preamble=0.

---

## 4. Фаза B - ML на размеченных чанках (2-4 недели)

> **Идея:** 10k Gemini-меток - не для «LLM на каждый КЗ», а для **узких моделей** offline.

### Активы

| Файл | Строк | Назначение |
|------|------:|------------|
| `data/ml/chunk_qa_fixes_wave_a.jsonl` | 10 000 | verdict, clean_text, entities |
| `data/ml/chunk_qa_issues.jsonl` | ~25 000 | weak_section_title, too_long… |
| `data/ml/chunk_qa_fixes_merged.jsonl` | merged | итог merge |
| `data/ml/feedback/*.jsonl` | растёт | retrieval_fix, methodist |

### B1. Export dataset (Cursor Auto, 1 день)

```bash
python3 scripts/export_training_feedback.py   # retrieval pairs
# TODO: scripts/export_chunk_qa_dataset.py → ml/datasets/chunk_qa_sft.jsonl
```

Формат строки:

```json
{"input": {"text": "...", "chunk_type": "...", "section_title": "..."},
 "output": {"verdict": "fix", "issues": ["weak_section_title"], "confidence": 0.91}}
```

### B2. Chunk Issue Classifier (локально, CPU)

- **Модель:** LightGBM или DistilBERT multi-label
- **Задача:** предсказать issues + verdict без Gemini (~5 ms/chunk)
- **Gate:** F1 по `preamble_leak`, `icd_inflation` ≥ 0.85
- **ROI:** Wave B/C chunk QA без GOOGLE $

```bash
pip install -r requirements-ml.txt
python3 ml/train/train_chunk_classifier.py --data ml/datasets/chunk_qa_sft.jsonl
```

### B3. Local Reranker LoRA (после D2)

```bash
python3 ml/train/finetune_reranker.py --dataset ml/datasets/retrieval_pairs.jsonl
python3 scripts/run_ab_embedder_kz.py
```

**Gate:** probe top-1 не хуже cloud embed; L2 latency −30%.

### B4. Интеграция

- `llm_chunk_qa.py`: если classifier confidence ≥ 0.9 → skip Gemini
- `rag_server.py`: `RAG_RERANK_BACKEND=local` после pass eval

### Что НЕ обучать

| ❌ | Почему |
|----|--------|
| LLM вместо 482 правил | send_gate должен быть детерминированным |
| Одна модель «на весь КЗ» | медленно, хуже правил на L1 |
| Fine-tune через Cursor API | Cursor не даёт train API |

---

## 5. Фаза C - слабые KZ и routing (Cursor Opus)

**Цель:** `report_n_1` ~60% → 75%+, `report_n_2` ~62% → 75%+

| Трек | Кошелёк | Действие |
|------|---------|----------|
| Scoring vs routing | Cursor Opus #1 | разбор batch JSON, gates в `consult_retrieval.py` |
| Sparse neuro caps | Auto + tests | seeds в `retrieval_fix.jsonl` |
| L2 lite на Render | env flags | evidence pack, align paths |

```bash
.venv/bin/python scripts/run_clients_consult_render_batch.py \
  --tier L2 --cases report_n_1,report_n_2 \
  --out ml/experiments/batch_report_n_r2/
```

---

## 6. Фаза D - continuous QA (GOOGLE, после gate)

```bash
.venv/bin/python scripts/build_chunk_qa_queue_tiered.py --max-total 500 \
  --out data/ml/chunk_qa_queue_continuous.jsonl
# Gemini только на queue, не Wave B 5k без gate
```

---

## 7. Операции: Telegram

```bash
# control loop (кнопки Да/Нет)
nohup bash scripts/telegram_control_loop.sh >> data/ml/reports/telegram_control.log 2>&1 &

# статус pipeline
nohup bash scripts/checklist_push_watchdog.sh loop >> data/ml/reports/checklist_notify.log 2>&1 &
```

`.env`: `TELEGRAM_*`, `TELEGRAM_NOTIFY_RENDER=0` (без сообщений про Render).

---

## 8. North star (метрики месяца)

| Метрика | Сейчас | Цель |
|---------|--------|------|
| Embed done | ~10k/98k | 100% + upload |
| Probe top-1 | 100% | не регрессировать |
| L2 smoke 5 | 85.6% | ≥85%, rag_chunks_n>0 |
| L1 batch | 84.2% | без регрессии; слабые рубрики ↑ после C |
| report_n_* | ~60% | 75%+ (routing track) |
| Classifier F1 P0 issues | - | ≥0.85 |
| Cursor $70 остаток | - | ≥$10 |

---

## 9. Порядок работ (чеклист)

```
☑ Wave A + prod r2 + B1-B3 code
☐ D2 embed 100%
☐ upload corpus + redeploy если нужно
☐ smoke post-embed
☐ export chunk_qa dataset
☐ train classifier v1
☐ reranker A/B
☐ report_n routing Opus session
☐ continuous queue 500 (optional)
```

---

## 10. Карта документов (что читать)

| Документ | Когда |
|----------|-------|
| **action-plan-master.md** (этот файл) | общий порядок работ |
| [cursor-spend-checklist.md](./cursor-spend-checklist.md) | куда тратить Cursor $70 |
| [chunk-quality-pipeline.md](./chunk-quality-pipeline.md) | chunk QA команды |
| [GEMINI_FULL_QA_PLAN.md](../ml/experiments/batch_r70_2026-06-28/GEMINI_FULL_QA_PLAN.md) | волны Gemini QA |
| [ml/README.md](../ml/README.md) | MLOps, feedback, train |
| [consult-l2-rebuild-plan.md](./consult-l2-rebuild-plan.md) | ускорение L2 |
| [architecture-b2c-patient.md](./architecture-b2c-patient.md) | B2C |
| [methodist-ml-priority-plan.md](./methodist-ml-priority-plan.md) | кабинет методиста |

---

## 11. Антипаттерны

| Не делать | Почему |
|-----------|--------|
| Ждать рост L1 от chunk QA | L1 без RAG |
| Wave B 5k без gate | сжигает GOOGLE $ |
| Opus на pytest typo | Auto |
| Deploy corpus без embed | JSONL на диске ≠ rerank |
| Дублировать ключи в .env | один ключ = один quota pool |
