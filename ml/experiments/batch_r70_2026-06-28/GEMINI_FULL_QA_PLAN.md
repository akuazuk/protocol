# План большого прогона Gemini: chunk QA до «идеала»

Дата: 2026-06-28 · после batch r70 (avg **84.2%**, 2 слабых KZ по safety/sparse)

## Цель

Точная разметка **каждого предложения** протокола: `chunk_type`, `indexable`, ICD, clinical labels, merge/split. Повторный прогон слабых чанков.

## Фаза 0 - baseline (30 мин)

```bash
.venv/bin/python scripts/audit_chunk_quality.py \
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
  --stats data/ml/reports/chunk_quality_2026-06-28.json

.venv/bin/python scripts/build_chunk_qa_queue.py --max 60000 \
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
  --out data/ml/chunk_qa_queue_full.jsonl
```

Критерии «плохой чанк» для retry:
- `quality_score < 0.72`
- issues: `preamble_leak`, `type_body_but_clinical`, `icd_inflation`, `empty_clinical`
- fix в `chunk_qa_fixes*.jsonl` с `action=drop` или пустой `chunk_type`

## Фаза 1 - протоколы из слабых KZ (1-2 ч)

Приоритетные PDF (routing + L2):
- `...нервной_системы_взр_нас_..._8.pdf`
- `...невромускular..._42.pdf` (2025)
- протоколы из `report_n_*` matched paths

```bash
.venv/bin/python scripts/build_kz_weak_chunk_qa_queue.py
# уже есть targeted queue; расширить --max-per-protocol 400
```

## Фаза 2 - полный Gemini без лимитов (8-12 ч)

```bash
export CHUNK_QA_LLM=1
export CHUNK_QA_LLM_BACKEND=gemini
export CHUNK_QA_MAX_OUT=16000
export CHUNK_QA_LLM_RETRIES=5

# 2a. targeted (weak KZ protocols) - retry все с --append
.venv/bin/python scripts/llm_chunk_qa.py \
  --queue data/ml/chunk_qa_queue_kz_targeted.jsonl \
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
  --out data/ml/chunk_qa_fixes_kz_targeted.jsonl \
  --batch-size 8 --append

# 2b. full queue - только чанки без fix или с низким score
.venv/bin/python scripts/llm_chunk_qa.py \
  --queue data/ml/chunk_qa_queue_full.jsonl \
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
  --out data/ml/chunk_qa_fixes_full.jsonl \
  --batch-size 8 --append
```

**Retry плохих прогонов:** удалить cache `data/ml/chunk_qa_cache/<chunk_id>.json` и строку fix из jsonl, затем `--append` снова.

## Фаза 3 - merge + dedup (15 мин)

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
by_id = {}
for fp in [
    "data/ml/chunk_qa_fixes.jsonl",
    "data/ml/chunk_qa_fixes_kz_targeted.jsonl",
    "data/ml/chunk_qa_fixes_full.jsonl",
]:
    p = Path(fp)
    if not p.is_file(): continue
    for line in p.open():
        r = json.loads(line); by_id[r["chunk_id"]] = r
Path("data/ml/chunk_qa_fixes_merged.jsonl").write_text(
    "\n".join(json.dumps(v, ensure_ascii=False) for v in by_id.values()) + "\n")
print("merged", len(by_id))
PY

.venv/bin/python scripts/merge_chunk_qa_fixes.py \
  --fixes data/ml/chunk_qa_fixes_merged.jsonl \
  --chunks output/rich_chunks/rich_chunks.final.jsonl

.venv/bin/python scripts/promote_rich_chunks_v2.py --source final
.venv/bin/python scripts/audit_chunk_quality.py \
  --chunks output/rich_chunks/rich_chunks.final.jsonl
```

Цель audit: `mean quality >= 0.94`, `icd_inflation == 0`, preamble &lt; 0.1%.

## Фаза 4 - Render + batch (1 ч)

```bash
./scripts/upload_rich_chunks_render.sh srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com --gzip
# Restart Render

.venv/bin/python scripts/run_clients_consult_render_batch.py \
  --tier L1 --kz-only --ai-review off \
  --out ml/experiments/batch_post_gemini_full/l1_kz
```

## Фаза 5 - если overall всё ещё &lt;85% на weak KZ

Chunk QA **не поднимет** `report_n_1` (dual NSAIDs) и **report_n_2** (sparse doc) - нужны:
- tuning `safety_checker` / NSAID combo rules
- sparse neurology caps review для коротких, но полных КЗ
- `retrieval_fix` golden set (уже seed в `data/ml/feedback/kz_routing_retrieval_fix_seed.jsonl`)

## Критерий «идеально»

| Уровень | Метрика |
|---------|---------|
| Corpus | quality mean ≥0.94, 0 preamble leak |
| L1 batch | avg ≥85%, weak KZ ≤1 |
| L2 weak | avg ≥70% на report_n_* |
| UI | criteria table без обрезки смысла (r71 col widths) |
