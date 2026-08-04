# План действий: targeted Gemini + улучшения по 6 слабым KZ

Дата: 2026-06-28 · BUILD: `2026-06-28-r67-kz-routing-gates`

> **B2C:** файлы `clients_consult/a_*.pdf` - лабораторные анализы, не КЗ. Исключены из метрик слабых KZ и targeted chunk QA.

## Контекст (что показали тесты)

### L1 batch — 29 файлов (26 КЗ + 3 B2C a_*)
- Средний overall **81.1%**, 6 кейсов &lt; 70%
- L1 на Render **не использует chunk-RAG** (`rag_used: false`)
- `rules_check` не отдаёт findings → batch не ловит false positive правил

### L2 batch — 3 слабых KZ (без a_* B2C)
| case | L1 | L2 | L2 matched protocol (проблема) |
|------|-----|-----|----------------------------------|
| report_n_1 | 60% | **35%** | nevrologiya **дет.** — routing |
| report_n_2 | 62% | 62% | nevrologiya **дет.** |
| A_2 | 62–70% | 62–70% | **ВИЧ** — явно неверный routing |
| | | | `rag_chunks_n: 0` в ответе (L2-lite) |

*(a_1, a_3, a_4 — B2C анализы, не входят в KZ-метрики.)*

**Вывод:** главный рычаг для слабых KZ — **routing протокола + L2 evidence**, не только качество чанков. Chunk QA нужен для протоколов, которые **ошибочно или слабо** попадают в RAG.

---

## Фаза 1 — Targeted Gemini (1–2 дня)

### 1.1 Очередь (готово)
```bash
.venv/bin/python scripts/build_kz_weak_chunk_qa_queue.py
# → data/ml/chunk_qa_queue_kz_targeted.jsonl (408 чанков)
# → data/ml/chunk_qa_queue_kz_targeted_manifest.json
```

Состав:
| priority | count | источник |
|----------|------:|----------|
| 130 | 32 | пропущенные Gemini (хвост 12k) |
| 128 | 4 | preamble в ВИЧ/невро протоколах |
| 126 | 4 | score &lt; 0.6 |
| 125 | 9 | chunk_qa_review |
| 120 | 359 | все чанки 2 протоколов L2 (до 200/doc) |

### 1.2 Прогон Gemini
```bash
CHUNK_QA_LLM=1 CHUNK_QA_LLM_BACKEND=gemini CHUNK_QA_MAX_OUT=8000 \
  .venv/bin/python scripts/llm_chunk_qa.py \
  --queue data/ml/chunk_qa_queue_kz_targeted.jsonl \
  --out data/ml/chunk_qa_fixes_kz_targeted.jsonl \
  --batch-size 5
```
Оценка: **~25–40 мин**, ~408 API calls.

### 1.3 Merge + promote + Render
```bash
# объединить fixes (dedup by chunk_id, targeted wins on conflict)
.venv/bin/python scripts/merge_chunk_qa_fixes.py \
  --fixes data/ml/chunk_qa_fixes_merged.jsonl   # см. скрипт merge ниже

.venv/bin/python scripts/promote_rich_chunks_v2.py --source final
./scripts/upload_rich_chunks_render.sh srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com --gzip
# Restart Render
```

**Скрипт merge fixes (разово):**
```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
by_id = {}
for fp in ["data/ml/chunk_qa_fixes.jsonl", "data/ml/chunk_qa_fixes_kz_targeted.jsonl"]:
    for line in Path(fp).open():
        r = json.loads(line); by_id[r["chunk_id"]] = r
Path("data/ml/chunk_qa_fixes_merged.jsonl").write_text(
    "\n".join(json.dumps(v, ensure_ascii=False) for v in by_id.values()) + "\n")
print("merged", len(by_id))
PY
# patch merge_chunk_qa_fixes.py DEFAULT_FIXES or pass env
```

### 1.4 Критерий приёмки
- [ ] 408/408 в `chunk_qa_fixes_kz_targeted.jsonl`
- [ ] `merge` → final.jsonl, audit score ≥ 0.936
- [ ] Upload Render, `/health` version ok
- [ ] Повтор L2 на 6 KZ: overall report_n_1 &gt; 45%, routing paths осмысленнее

---

## Фаза 2 — Движок КЗ (3–5 дней, код)

### 2.1 Routing протокола (P0)
**Проблема:** a_* → ВИЧ, report_n_* → детская невrologия без проверки аудитории/МКБ.

| Задача | Файл | Действие |
|--------|------|----------|
| Audience gate | `consult_retrieval.py`, `consult_alignment.py` | если КЗ **взрослый**, штраф **дет.** протоколов |
| Rubric sanity | `consult_analysis.py` | если нет маркеров ВИЧ/ART — исключить ВИЧ-КП из top-3 |
| retrieval_fix seed | `data/ml/feedback/retrieval_fix.jsonl` | 6 записей из L2 batch (ручной chosen_path) |

### 2.2 Sparse / short KZ (P0)
**Проблема:** a_*, report_n_2 → 61.9% `low_confidence`.

| Задача | Действие |
|--------|----------|
| Gate `text_len < 1500` | status `insufficient_text`, не считать compliance |
| OCR fallback | `text_extract.py` — warn в report_markdown |

### 2.3 report_n_1 treatment/safety (P1)
- L1: treatment 25%, safety 0%, critical=3
- Разбор `score_breakdown` vs текст КЗ → калибровка блоков или rule skip для neurology template

### 2.4 rules_check на Render L1 (P1)
- Включить `clinical_rules` в L1 tier response (сейчас `rules_pct: null`)
- Иначе batch не регрессирует

### 2.5 protocol_applicability cap (P2)
- Дефолт 90% на всех кейсах → мало дискриминации; привязать к `alignment_mean_score`

---

## Фаза 3 — ML (2–4 недели)

### 3.1 Не делать сейчас
- Fine-tune LLM на метках чанков (`drop`/`merge`) — мало связи с overall KZ
- Полный rebuild 59k embed offline без retrieval_fix

### 3.2 Делать после 6+ retrieval_fix
```bash
# 1. Экспорт пар из methodist feedback
.venv/bin/python scripts/export_retrieval_pairs.py  # если есть

# 2. LoRA embedder
.venv/bin/python ml/train/finetune_embedder.py \
  --dataset ml/datasets/retrieval_pairs_resolved.jsonl

# 3. A/B
.venv/bin/python scripts/run_ab_embedder_kz.py
```

### 3.3 Калибратор overall (опционально)
- Фичи: `text_len`, block scores, `matched_protocols_count`, rubric
- Target: methodist override / AI rating
- 29 KZ мало → собрать ещё 20–30 из clients_consult + feedback

---

## Фаза 4 — Контроль качества (регулярно)

```bash
# Полный L1 Render batch
.venv/bin/python scripts/run_clients_consult_render_batch.py --ai-review auto

# L2 только слабые
.venv/bin/python scripts/run_clients_consult_render_batch.py --tier L2 --cases report_n_1 a_1 ...

# Nightly chunk audit
.venv/bin/python scripts/nightly_chunk_quality_report.py
```

---

## Приоритетный backlog (топ-7)

1. **Gemini targeted 408** → merge → Render
2. **retrieval_fix** для 6 KZ (wrong_protocol)
3. **Audience/rubric gates** в consult_retrieval
4. **Sparse KZ gate** (61.9% trap)
5. **rules_check в L1** на Render
6. **Повтор batch** L1+L2 на 6 KZ
7. **LoRA embedder** после ≥10 retrieval_fix

---

## Файлы артефактов

| Файл | Назначение |
|------|------------|
| `ml/experiments/batch_clients_consult_2026-06-28/report.json` | L1 29 KZ |
| `ml/experiments/batch_clients_consult_2026-06-28/l2_weak_report.json` | L2 6 KZ |
| `ml/experiments/batch_clients_consult_2026-06-28/REPORT.md` | сводка L1 |
| `data/ml/chunk_qa_queue_kz_targeted.jsonl` | очередь Gemini |
| `scripts/build_kz_weak_chunk_qa_queue.py` | пересборка очереди |
| `scripts/run_clients_consult_render_batch.py` | повтор batch |
