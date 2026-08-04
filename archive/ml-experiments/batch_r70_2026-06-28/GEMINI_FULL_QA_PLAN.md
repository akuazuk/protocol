# План Gemini chunk QA (corpus-wide): без лишних часов

Дата: 2026-06-28 · корпус `rich_chunks.final.jsonl` · batch KZ r70 avg **84.2%**

## 1. Зачем этот прогон и чего он **не** делает

**Цель:** улучшить **RAG по всем 478 протоколам / 24 рубрикам** - точнее `chunk_type`, ICD, entities, merge/split, меньше preamble в выдаче.

**Не цель:** поднять overall в batch KZ там, где виноват **scoring** (dual NSAIDs, sparse caps), а не чанки. Это отдельная ветка (safety/sparse), не Gemini.

**Не использовать:** `build_kz_weak_chunk_qa_queue.py` и `report_n_*` - это legacy targeted под 2 тестовых KZ; **407/407 уже закрыты**.

---

## 2. Текущее состояние (не гонять заново)

| Артефакт | Значение |
|----------|----------|
| Чанков в corpus | **59 045** |
| Рубрик в corpus | **24** (от stomatologiya до akusherstvo…) |
| Уже в fixes merged | **~12 170** chunk_id |
| Mean quality (audit) | **0.936**, median **1.0** |
| Flagged audit (любой issue) | **26 772** - **не равно** «нужен Gemini» |

**Вывод:** повторно гонять все flagged = **~15-20 ч впустую**. Нужна **tier-очередь** + **исключение уже исправленных**.

---

## 3. Принципы очереди (ROI)

### 3.1 В Gemini **отправляем** (влияет на consult/L2 RAG)

| Приоритет | Issue / условие | Почему |
|-----------|-----------------|--------|
| **P0** | `preamble_leak`, `icd_inflation` | Мусор и ложный ICD в retrieval |
| **P0** | `type_body_but_clinical` | Клиника не находится (тип body) |
| **P1** | `truncated_list` + тип `diagnostics`/`treatment`/`drug_list`/`criteria_block` | Обрыв алгоритма/списка |
| **P1** | `too_long` + clinical type | Кандидат split/merge |
| **P1** | `empty_entities` + clinical type + score &lt; 0.75 | Пустые lab/drug при клиническом тексте |
| **P2** | score &lt; 0.65 (любой тип) | Явно слабый чанк |
| **P2** | `retrieval_fix` / feedback `source_path` | Протоколы с разметкой методиста |

### 3.2 В Gemini **не отправляем** (экономия ~60% очереди)

| Пропуск | Причина |
|---------|---------|
| chunk_id уже в `chunk_qa_fixes_merged.jsonl` с `confidence ≥ 0.85` | Уже обработан |
| Только `weak_section_title`, score ≥ **0.88** | Косметика заголовка, RAG по text OK |
| Только `too_short` в `terms`/`appendix` | Служебные блоки |
| `chunk_type=body`, score ≥ 0.9, нет P0 issues | Низкий ROI |
| Random 2% sample | Уже было в первом 12k прогоне |

### 3.3 Покрытие специальностей и МКБ (не через weak KZ)

**Матрица покрытия** - в очередь обязательно попадает:

1. **Каждая из 24 рубрик** - минимум **N чанков P0/P1** (стратификация по `source_path`).
2. **Клинические типы** - доля в очереди ≥ доли в corpus: `treatment`, `diagnostics`, `drug_list`, `criteria_block`.
3. **МКБ из пилотных KZ** - для каждого PDF в `clients_consult/` (30 KZ, все специальности: gastro, kard, pediatr, lor, uro, aler, ter, …):
   - извлечь ICD из КЗ → найти `source_path` matched протоколов (L1 batch / `protocol_cards`) → **все P0/P1 чанки этих PDF** в wave 1 (не только невро).

B2C `A/a*` в batch **не учитываем** как KZ.

---

## 4. Фазы (с gate - не идём дальше, если gate не пройден)

### Фаза 0 - Baseline и tier-очередь (~45 мин)

```bash
CHUNKS=output/rich_chunks/rich_chunks.final.jsonl

.venv/bin/python scripts/audit_chunk_quality.py \
  --chunks "$CHUNKS" \
  --stats data/ml/reports/chunk_quality_baseline.json \
  --report data/ml/reports/chunk_quality_baseline.md

# Tier-очередь (см. скрипт ниже) - НЕ --max 60000 слепо
.venv/bin/python scripts/build_chunk_qa_queue_tiered.py \
  --chunks "$CHUNKS" \
  --fixes data/ml/chunk_qa_fixes_merged.jsonl \
  --kz-folder clients_consult \
  --out data/ml/chunk_qa_queue_tiered.jsonl \
  --manifest data/ml/chunk_qa_queue_tiered_manifest.json
```

**Ожидаемый размер tier-очереди:** **~6 000-10 000** чанков (не 27k).

Manifest должен показать: `by_rubric`, `by_priority`, `by_clinical_type`, `kz_icd_protocol_paths`.

---

### Фаза 1 - Pilot **800** чанков (~1-1.5 ч)

```bash
export CHUNK_QA_LLM=1 CHUNK_QA_LLM_BACKEND=gemini
export CHUNK_QA_MAX_OUT=16000 CHUNK_QA_LLM_RETRIES=5

.venv/bin/python scripts/llm_chunk_qa.py \
  --queue data/ml/chunk_qa_queue_tiered.jsonl \
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
  --out data/ml/chunk_qa_fixes_pilot.jsonl \
  --limit 800 --batch-size 8
```

**Gate pilot (обязательно):**

```bash
.venv/bin/python scripts/merge_chunk_qa_fixes.py \
  --fixes data/ml/chunk_qa_fixes_pilot.jsonl \
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
  --out output/rich_chunks/rich_chunks.pilot.jsonl

.venv/bin/python scripts/audit_chunk_quality.py \
  --chunks output/rich_chunks/rich_chunks.pilot.jsonl \
  --stats data/ml/reports/chunk_quality_pilot.json
```

| Критерий | Порог | Если нет |
|----------|-------|----------|
| Applied fixes / pilot | ≥ **70%** | Смотреть `chunk_qa_review.jsonl`, править prompt, **не** wave 2 |
| `preamble_leak` delta | ↓ ≥ **50%** на pilot set | Ужесточить drop rules |
| `icd_inflation` | **0** новых после merge | Стоп, разбор merge |
| Случайная выборка 20 fix | ≥ **16/20** осмысленны вручную | Стоп |

**Только после gate** - wave 2.

---

### Фаза 2 - Wave A: P0 + KZ-linked protocols (~4-6 ч)

Остаток очереди с `priority ≥ 90` + все чанки PDF из `kz_icd_protocol_paths` в manifest.

```bash
.venv/bin/python scripts/llm_chunk_qa.py \
  --queue data/ml/chunk_qa_queue_tiered.jsonl \
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
  --out data/ml/chunk_qa_fixes_wave_a.jsonl \
  --batch-size 8 --append
```

`--append` + skip cached: уже обработанные pilot chunk_id не дублируются.

---

### Фаза 3 - Wave B: P1 только если Wave A дала delta (~3-5 ч)

Условие старта:

- audit mean quality **≥ 0.938** после wave A (было 0.936), **или**
- L2 smoke на **5 KZ из разных рубрик** (gastro_1, kard_1, pediatr_1, report_lor_1, report_urolog_1): `rag_chunks_n > 0` и цитаты без preamble.

Иначе **Wave B не запускаем** - сначала merge/review wave A.

Фильтр wave B: `priority 70-89`, cap **5000** чанков.

---

### Фаза 4 - Retry плохих fix (~30-90 мин, не full rerun)

Только chunk_id где:

- fix `confidence < 0.6` или `verdict=ok` при issues P0;
- пустой `corrected_chunk_type` при `type_body_but_clinical`;
- запись в `chunk_qa_review.jsonl`.

```bash
# Сброс cache для списка retry_ids.txt, затем:
.venv/bin/python scripts/llm_chunk_qa.py \
  --queue data/ml/chunk_qa_retry.jsonl \
  --out data/ml/chunk_qa_fixes_retry.jsonl \
  --batch-size 4 --append
```

**Не** перегонять весь corpus.

---

### Фаза 5 - Merge, promote, Render (~30 мин)

```bash
# merge все waves (pilot + a + b + retry + legacy)
python3 - <<'PY'
import json
from pathlib import Path
by_id = {}
for fp in [
    "data/ml/chunk_qa_fixes_merged.jsonl",
    "data/ml/chunk_qa_fixes_pilot.jsonl",
    "data/ml/chunk_qa_fixes_wave_a.jsonl",
    "data/ml/chunk_qa_fixes_wave_b.jsonl",
    "data/ml/chunk_qa_fixes_retry.jsonl",
]:
    p = Path(fp)
    if not p.is_file():
        continue
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        by_id[r["chunk_id"]] = r
Path("data/ml/chunk_qa_fixes_merged.jsonl").write_text(
    "\n".join(json.dumps(v, ensure_ascii=False) for v in by_id.values()) + "\n")
print("merged", len(by_id))
PY

.venv/bin/python scripts/merge_chunk_qa_fixes.py \
  --fixes data/ml/chunk_qa_fixes_merged.jsonl \
  --chunks output/rich_chunks/rich_chunks.final.jsonl

.venv/bin/python scripts/promote_rich_chunks_v2.py --source final

.venv/bin/python scripts/audit_chunk_quality.py \
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
  --baseline data/ml/reports/chunk_quality_baseline.json
```

**Цели corpus (realistic):**

| Метрика | Было | Цель |
|---------|------|------|
| mean quality | 0.936 | **≥ 0.945** |
| preamble_leak | 642 | **&lt; 100** |
| type_body_but_clinical | 4043 | **&lt; 1500** |
| icd_inflation | ~0 | **0** |

---

### Фаза 6 - Приёмка на KZ (все специальности, ~40 мин)

```bash
.venv/bin/python scripts/run_clients_consult_render_batch.py \
  --tier L1 --kz-only --ai-review off \
  --out ml/experiments/batch_post_gemini_tiered/l1_kz

# L2 выборочно: 1 KZ на рубрику из clients_consult (не report_n only)
.venv/bin/python scripts/run_clients_consult_render_batch.py \
  --tier L2 --cases gastro_1,kard_1,pediatr_1,report_lor_1,report_urolog_1 \
  --out ml/experiments/batch_post_gemini_tiered/l2_sample
```

**Успех chunk QA:**

- L1 avg **≥ 84%** (не хуже r70) - chunk QA не должен ломать scoring;
- L2 sample: **rag_used / citations** лучше или не хуже r70;
- **не** требуем avg 95% на batch - это смешивает scoring и RAG.

**Если L1 упал** - rollback corpus на предыдущий `rich_chunks.final.jsonl` backup, разбор review.

---

## 5. Оценка времени (реалистично)

| Этап | Чанков | Время |
|------|--------|-------|
| Фаза 0 queue | - | **45 мин** |
| Pilot + gate | 800 | **1-1.5 ч** |
| Wave A | ~4 000-6 000 | **4-6 ч** |
| Wave B (optional) | ~3 000-5 000 | **3-5 ч** |
| Retry | ~200-800 | **0.5-1.5 ч** |
| Merge + audit + batch | - | **1.5 ч** |
| **Итого (без Wave B)** | **~5-8k** | **~8-10 ч** |
| **Итого (с Wave B)** | **~8-12k** | **~12-15 ч** |

Слепой прогон **27k flagged** = **18-25 ч** с **низким ROI** - **не делаем**.

---

## 6. Чем этот план лучше предыдущего прогона

| Было (12k bulk) | Станет (tiered) |
|-----------------|-----------------|
| Top-N по эвристике, без gate | **Pilot 800 + gate** перед масштабом |
| `weak_section_title` раздувает очередь | **Исключены** одиночные weak title |
| Повтор уже исправленных chunk_id | **Skip** merged fixes ≥0.85 |
| Привязка к weak KZ / невро | **24 рубрики + ICD из всех KZ** в clients_consult |
| Один merge → надежда | **Delta audit** после каждой wave |
| MAX_OUT 4000 | **16000** + retry только bad fix |
| Критерий «avg 95% batch» | **Раздельно:** corpus metrics + L1 не хуже + L2 citations |

---

## 7. Стоп-условия (часы не тратить)

1. Pilot gate не пройден → **стоп**, правка prompt/merge, не wave A.
2. Wave A: mean quality +0.002 → **не** wave B.
3. Geo-block Gemini &gt; **5%** batch → пауза, retry queue, не продолжать слепо.
4. Batch L1 avg **&lt; 82%** после deploy → **rollback** corpus.
5. Два слабых KZ (`report_n_1`, `report_n_2`) **не** критерий отмены chunk QA - там scoring; фиксить **safety/sparse**, не второй full Gemini.

---

## 8. Скрипт tier-очереди

`scripts/build_chunk_qa_queue_tiered.py` - реализует §3 и §3.3:

```bash
.venv/bin/python scripts/build_chunk_qa_queue_tiered.py \
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
  --fixes data/ml/chunk_qa_fixes_merged.jsonl \
  --kz-folder clients_consult \
  --max-total 10000 --min-per-rubric 5
```

Manifest: `data/ml/chunk_qa_queue_tiered_manifest.json` - проверить `rubrics_covered` (цель **24**) и `queue_size` (**6-10k**, не 27k).

После manifest OK - **Фаза 1 pilot 800**.

---

## 9. Связанные файлы

| Файл | Назначение |
|------|------------|
| `scripts/llm_chunk_qa.py` | Gemini batch |
| `scripts/merge_chunk_qa_fixes.py` | apply fixes |
| `scripts/audit_chunk_quality.py` | метрики до/после |
| `data/ml/chunk_qa_fixes_merged.jsonl` | уже обработанные id |
| `clients_consult/` | 30 KZ всех специальностей для ICD→protocol mapping |
| `ml/experiments/batch_r70_2026-06-28/REPORT.md` | baseline batch |
