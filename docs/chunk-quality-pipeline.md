# Техническое задание: конвейер качества rich-чанков

Версия: 2026-06-24  
Статус: реализуется  
Корпус: 478 PDF, ~64k чанков в `output/rich_chunks/rich_chunks.jsonl`

## 1. Цель

Повысить качество retrieval и consult-review за счёт:

- чистого текста чанков (без преамбулы и колонтитулов);
- корректных `chunk_type`, `tags`, `indexable`;
- ICD только из текста чанка (протокол - отдельным полем);
- rule-based QA с отчётами;
- выборочного LLM-QA offline (не в hot path).

Ожидаемый эффект: recall@5 +5-10%, preamble в top-10 → 0, меньший индекс.

## 2. Архитектура конвейера

```
PDF → build_rich_chunks.py (v2)
    → rich_chunks.jsonl
    → enrich_rich_chunk_tags.py
    → apply_chunk_rule_fixes.py → rich_chunks.v2.jsonl
    → audit_chunk_quality.py → reports + chunk_qa_issues.jsonl
    → build_chunk_qa_queue.py → chunk_qa_queue.jsonl
    → llm_chunk_qa.py (offline) → chunk_qa_fixes.jsonl
    → merge_chunk_qa_fixes.py → rich_chunks.final.jsonl
    → build_chunk_embeddings.py → RAG index
```

Runtime: `rag_server.py` загружает JSONL, пропускает `indexable: false`, использует `tags.signal`.

## 3. Модули и контракты

### 3.1 `clinical_knowledge/chunk_quality.py`

| Функция | Назначение |
|---------|------------|
| `ISSUE_*` | Коды проблем: `icd_inflation`, `weak_section_title`, `type_body_but_clinical`, `too_short`, `too_long`, `preamble_leak`, `empty_entities`, `truncated_list` |
| `detect_issues(chunk) -> list[str]` | Список issue-кодов |
| `quality_score(chunk) -> float` | 0.0-1.0 |
| `should_index(chunk) -> bool` | Индексировать в RAG |
| `strip_noise_lines(text) -> str` | Убрать колонтитулы |
| `fix_weak_section_title(chunk, parent_title) -> str` | Нормализация заголовка |
| `is_icd_inflation(chunk) -> bool` | 15+ ICD при text < 200 |
| `apply_indexable_flags(chunk) -> dict` | Добавляет `indexable`, `noise_flags` |

### 3.2 `clinical_knowledge/chunk_type_infer.py`

| Функция | Назначение |
|---------|------------|
| `infer_chunk_type(section_title, section_number, section_path, text) -> str` | Приоритет: номер раздела → section_path → regex по полному тексту → `body` |
| `SECTION_TYPE_PATTERNS` | Маппинг «1. Диагностика» → `diagnostics` |

### 3.3 `clinical_knowledge/chunk_entities.py`

| Функция | Назначение |
|---------|------------|
| `extract_lab_tests_enriched(text)` | LAB_RE + синонимы (ОАК, коагулограмма) |
| `extract_imaging_enriched(text)` | IMAGING + РОГП, НСГ, ЭхоКГ |
| `enrich_chunk_entities(chunk) -> dict` | Обновляет lab_tests, imaging, drugs |

### 3.4 `clinical_knowledge/chunk_qa_schema.py`

Pydantic/dataclass схема ответа LLM:

```json
{
  "chunk_id": "str",
  "verdict": "ok|fix|drop|merge_with_next",
  "corrected_chunk_type": "str|null",
  "corrected_section_title": "str|null",
  "clean_text": "str|null",
  "obligation": "required|recommended|optional|contraindicated|null",
  "entities": {"exam": [], "drug": [], "condition": []},
  "noise_reasons": [],
  "confidence": 0.0,
  "notes": "str"
}
```

### 3.5 `clinical_knowledge/chunk_qa_prompt.py`

- `SYSTEM_CHUNK_QA` - system prompt (русский, без выдумывания фактов)
- `build_chunk_qa_prompt(chunks_batch, protocol_title) -> str`
- `build_protocol_sections_prompt(full_outline) -> str`

## 4. Скрипты

### 4.1 `scripts/audit_chunk_quality.py`

```bash
.venv/bin/python scripts/audit_chunk_quality.py \
  [--chunks PATH] [--out-jsonl PATH] [--report PATH] [--limit N]
```

Выход: JSON stats + markdown report + `chunk_qa_issues.jsonl` (chunks с score < 0.7 или issues).

### 4.2 `scripts/apply_chunk_rule_fixes.py`

```bash
.venv/bin/python scripts/apply_chunk_rule_fixes.py \
  --in output/rich_chunks/rich_chunks.jsonl \
  --out output/rich_chunks/rich_chunks.v2.jsonl \
  --fixes data/ml/chunk_rule_fixes.jsonl
```

Операции: `set_chunk_type`, `set_indexable_false`, `trim_text`, `fix_section_title`, `rebuild_embedding`, `merge_short_chunks`, `set_icd10_text_only`.

### 4.3 `scripts/build_chunk_qa_queue.py`

Приоритет: feedback → score < 0.5 → body+clinical → clinical types → random 2%.

### 4.4 `scripts/llm_chunk_qa.py`

Env: `CHUNK_QA_LLM=1`, кэш `data/ml/chunk_qa_cache/{hash}.json`.  
Batch 5-10 чанков на протокол. Флаги: `--queue`, `--out`, `--limit`, `--resume`, `--doc-id`.

### 4.5 `scripts/llm_protocol_sections_qa.py`

Document-level: 478 вызовов → `data/ml/protocol_section_map/{doc_id}.json`.

### 4.6 `scripts/merge_chunk_qa_fixes.py`

- `confidence >= 0.85` + fix → auto-apply
- `drop` → `indexable: false`
- `0.6-0.85` → `chunk_qa_review.jsonl`

### 4.7 `scripts/nightly_chunk_quality_report.py`

Cron-обёртка над audit + сравнение с baseline JSON.

### 4.8 `scripts/ingest_retrieval_feedback_to_qa_queue.py`

Читает `data/ml/feedback/*.jsonl` → дополняет queue.

### 4.9 `scripts/enrich_rich_chunk_tags.py` (доработка)

Флаги: `--dry-run`, `--stats-out`.

## 5. Изменения `build_rich_chunks.py` (v2)

1. `infer_chunk_type()` вместо `guess_chunk_type()`.
2. Поля: `icd10_protocol`, `icd10_codes` (только из текста), `indexable`, `noise_flags`.
3. `build_embedding_ready_text()` - ICD из текста чанка; для overview - протокол.
4. Post-filter: preamble blocks → `indexable: false`.
5. `strip_noise_lines()` на text перед записью.
6. `enrich_chunk_entities()` для lab/imaging.
7. Tags через `build_chunk_tags()` при сборке (уже есть).

## 6. Runtime (`rag_server.py`, `rich_chunk_search.py`)

- `should_skip_rich_chunk_row`: + проверка `indexable is False`
- `RAG_CHUNKS_JSONL` может указывать на `rich_chunks.v2.jsonl`

## 7. Схема чанка (расширенная)

```json
{
  "chunk_id": "...",
  "chunk_type": "diagnostics",
  "text": "...",
  "embedding_ready_text": "...",
  "icd10_codes": ["D68.6"],
  "icd10_protocol": ["D68.6", "B24"],
  "indexable": true,
  "noise_flags": [],
  "quality_score": 0.91,
  "tags": {
    "signal": "high",
    "obligation": "required",
    "clinical_intent": "diagnose",
    "care_setting": ["ambulatory"],
    "entities": {"exam": ["ОАК"]},
    "is_preamble": false
  }
}
```

## 8. Метрики acceptance

| Метрика | Baseline | Цель v1 (rule) | Цель v2 (+LLM) |
|---------|----------|----------------|----------------|
| tags заполнены | 0% | 100% | 100% |
| body + clinical | 8.9% | <5% | <2% |
| short <80 | 8.2% | <5% | <3% |
| long >1200 | 9.2% | <4% | <3% |
| ICD inflation | ~5000 | <1000 | <500 |
| indexable=false | 0 | 2-4k | 3-4k |

## 9. Тесты

- `tests/test_chunk_quality.py` - issue detection, should_index
- `tests/test_chunk_type_infer.py` - section-based types
- `tests/fixtures/chunks.golden.jsonl` - 10 эталонных чанков
- Расширить `tests/test_rich_chunk_search.py` - skip indexable=false

## 10. Фазы и календарь

| Фаза | Срок | Deliverable |
|------|------|-------------|
| 1 Baseline | 2-3 д | tags, audit, baseline report |
| 2 Rule fixes in builder | 5-7 д | build_rich_chunks v2, rebuild pilot |
| 3 Post-process | 3-4 д | apply_chunk_rule_fixes, runtime |
| 4 LLM QA | 7-10 д | queue, llm_chunk_qa, merge |
| 5 Deploy | 3-4 д | embeddings, probe, nightly |

## 11. Риски

| Риск | Mitigation |
|------|------------|
| LLM галлюцинации | auto-apply только confidence >= 0.85; diff ratio clean_text |
| Merge ломает page_from | min page_from, max page_to |
| Полный rebuild долго | `--resume`, pilot rubric |
| Регресс retrieval | golden chunks + probe baseline |

## 12. Команды быстрого старта

```bash
# Tags
.venv/bin/python scripts/enrich_rich_chunk_tags.py

# Rule fixes на существующем корпусе
.venv/bin/python scripts/apply_chunk_rule_fixes.py

# Audit
.venv/bin/python scripts/audit_chunk_quality.py

# LLM queue (offline)
CHUNK_QA_LLM=1 .venv/bin/python scripts/llm_chunk_qa.py --limit 100

# Тесты
.venv/bin/python -m pytest tests/test_chunk_quality.py tests/test_chunk_type_infer.py -q
```
