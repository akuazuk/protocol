# Cursor $70: как тратить с максимальным ROI (Protocol)

> **Продукт:** проверка **любого** КЗ **любой** специальности против **478** протоколов Минздрава РБ (B2B врач + B2C пациент).  
> **Не путать:** десятки тысяч КЗ в год на рынке ≠ один batch из 30 fixtures — batch **измеряет** pipeline, chunk QA **улучшает** retrieval для всех.

---

## 0. Статус (2026-06-29, post-Wave A)

| Артефакт | Факт |
|----------|------|
| Wave A | **10 000** fixes, merged **16 577** total |
| Corpus prod | **57 852** chunks, BUILD `2026-06-29-r1-corpus-deploy-fix` |
| L1 KZ (30) post-deploy | **84.2%** avg - **без изменений** vs r70 |
| L2 smoke (5) | **85.6%** avg - без изменений |
| Search probe (100) | top-1 **100%** |
| Слабые KZ | `report_n_1` ~60%, `report_n_2` ~62% - **scoring/routing**, не чанки |

**Урок:** Wave A на prod **до** routing B1 дал corpus hygiene, но **не** поднял L1 overall. L1 tier на Render **не использует chunk-RAG** (`rag_used: false`) - chunk QA влияет на **L2, search, B2C crosscheck**, не на L1 batch напрямую.

**Следующий порядок (не откатывать Wave A, дожать эффект):**

1. **$0** - свежий audit 57k + `build_chunk_embeddings.py` + локальный symptom probe
2. **Cursor Auto** - B1 routing PR (audience + rubric gates)
3. **Cursor Auto** - B2 три P0 из §17 B2C
4. **$0** - L2 smoke с проверкой `rag_chunks_n` и preamble в JSON ответа
5. **GOOGLE** - только retry fix + continuous queue из feedback, **не** Wave B без gate

Подробности: [`data/ml/reports/chunk_qa_progress.md`](../data/ml/reports/chunk_qa_progress.md), batch `ml/experiments/batch_post_deploy_2026-06-29/`.

---

## 1. Главное за 60 секунд

### Что дают $70 Cursor (Pro+)

| Да | Нет |
|----|-----|
| 3–5 **узких** Agent-сессий с **Opus/Sonnet** (выбор модели вручную) | Прогон 59k чанков |
| Архитектура, приоритеты, multi-file fix **после** метрик | Замена `llm_chunk_qa.py` |
| «Какой PR делать первым» | Бесплатный unlimited Auto |

**Auto + Tab** на paid-плане — **без** этого пула. **Рутину кодить в Auto**, Opus — только synthesis и сложный routing.

### Где реальный прирост качества КZ (весь проект)

| Приоритет | Рычаг | Кошелёк | Что реально двигает |
|-----------|--------|---------|---------------------|
| **1** | **Embeddings + audit** после merge corpus | **$0** | RAG/rerank на prod читает новые поля чанков |
| **2** | **Routing / retrieval** (audience, rubric, sparse) | **$0 код + Cursor $** | ↑ L2, `report_n_*`, wrong PDF path |
| **3** | Chunk QA (pilot → gate → wave A) | **GOOGLE_API_KEY** | ↑ search, L2 citations, B2C crosscheck; **не L1 overall** |
| **4** | **3 quick wins B2C** в код | **Auto** | ↑ конверсия «понял → пришёл с вопросами» |
| **5** | B2B→B2C sanitization | **Auto** | ↓ риск утечки send_gate/ЦИСЗ |

**Вывод:** после Wave A $70 Cursor **оптимальны** на **п.2 и п.4–5**, не на повторный bulk Gemini. **Неоптимально:** Opus + «улучши чанки» или Wave B без gate.

### North star (как понять, что деньги не зря)

| Метрика | Baseline (r70) | Post-Wave A (29.06) | Цель месяца |
|---------|----------------|---------------------|-------------|
| Chunk mean quality | ~0.936 | **пере-audit 57k** (TBD) | ≥0.938 **и** embeddings rebuilt |
| Batch KZ overall (L1) | ~84.2% | **84.2%** (без регрессии) | слабые рубрики ↑ **после B1** |
| L2 smoke 5 рубрик | часть с `rag_chunks_n=0` | overall 85.6%; **rag/preamble не в batch JSON** | все 5: `rag_chunks_n>0`, preamble=0 |
| Search probe top-1 | baseline | **100%** (100/100) | не регрессировать |
| `report_n_1/2` | ~60–62% | без изменений | **scoring track**, не chunk QA |
| B2C pytest matrix | neuro/derma/phleb pass | - | +1 fixture после B2 |
| Cursor $70 остаток | — | — | ≥$10 в конце месяца |

---

## 2. Три кошелька

```
┌─────────────────────────────────────────────────────────┐
│  GOOGLE_API_KEY  →  chunk QA, L2/consult на Render      │
│  $0              →  audit, tier queue, pytest, merge    │
│  Cursor ~$70     →  2× Opus-диагностика + Auto-код      │
└─────────────────────────────────────────────────────────┘
```

---

## 3. План месяца (один лист — отмечай ☐)

> **Жёсткое правило (добавлено после 29.06):** prod deploy Wave A **желательно после B1** или явного waiver. Если Wave A уже на prod - **сначала D1–D2**, потом B1, потом приёмка.

### Неделя A — корпус (без Cursor $) — ☑ в основном сделано

| ☐ | Действие | Команда / артеfact |
|---|----------|-------------------|
| A0 | **Section map** 478 PDF (если ещё нет) | `scripts/llm_protocol_sections_qa.py` → `data/ml/protocol_section_map/` |
| A1 | Baseline audit | `scripts/audit_chunk_quality.py` → `chunk_quality_baseline.json` |
| A2 | Tier-очередь **24 рубрики** + ICD | `scripts/build_chunk_qa_queue_tiered.py` |
| A3 | Регрессия | `pytest tests/test_consult_retrieval.py tests/test_patient_*case*.py … -q` |
| A4 | **Pilot 800** | `scripts/llm_chunk_qa.py --limit 800` (GOOGLE) |
| A5 | **Gate pilot** | merge + audit; см. §4 |

### Неделя B — Cursor $ (**сейчас главный фокус**)

| ☐ | Прогон | Модель | ~$ | Deliverable |
|---|--------|--------|-----|-------------|
| B1 | **Routing root-cause** (все рубрики) | Auto → **1× Opus** | 20–25 | Таблица + 1–2 PR с тестом |
| B2 | **B2C: 3 quick wins** | Auto implement | 10–15 | 3 коммита + pytest |
| B3 | Sanitization boundary | Auto | 5–8 | ≤5 рисков или зелёный pytest |
| — | Резерв | Auto | 10+ | Фиксы после smoke |

### Неделя C — корпус на prod (GOOGLE) — ☑ Wave A + deploy

| ☐ | Wave A P0 + KZ-linked | `llm_chunk_qa.py --append` |
| ☐ | merge из **`section_mapped`**, не пустого `final` | `merge_chunk_qa_fixes.py` |
| ☐ | promote + upload Render | `upload_rich_chunks_render.sh … --gzip` |
| ☐ | **Embeddings rebuild** | `scripts/build_chunk_embeddings.py` |
| ☐ | L2 smoke **5 рубрик** + поля в JSON | см. §7 |
| ☐ | Batch snapshot | `run_clients_consult_render_batch.py` |

### Неделя D — дожать Wave A (**следующие 1–2 дня, $0 + Cursor B**)

| ☐ | Действие | Команда / критерий |
|---|----------|-------------------|
| D1 | Post-deploy audit 57k | `audit_chunk_quality.py` на `rich_chunks.final.jsonl` |
| D2 | Embeddings на prod/local | `build_chunk_embeddings.py`; gate: probe не хуже baseline |
| D3 | B1 routing PR | audience + rubric gates → merge |
| D4 | Повтор L2 smoke + 6 слабых KZ | `report_n_*` - отдельный scoring, не отмена chunk QA |
| D5 | Continuous queue | feedback → `build_chunk_qa_queue_tiered.py --max-total 500` |

---

## 4. Gate pilot (обязательный стоп-кран)

После pilot 800 **не** wave A и **не** Opus «улучши чанки», пока:

| Критерий | Порог |
|----------|-------|
| Applied fixes / pilot | ≥ 70% |
| `preamble_leak` на pilot set | ↓ ≥ 50% |
| `icd_inflation` после merge | **0** новых |
| Ручная выборка 20 fix | ≥ 16/20 |
| Рубрик в pilot | ≥ **8** из 24 |

```bash
.venv/bin/python scripts/merge_chunk_qa_fixes.py \
  --fixes data/ml/chunk_qa_fixes_pilot.jsonl \
  --chunks output/rich_chunks/rich_chunks.section_mapped.jsonl \
  --out output/rich_chunks/rich_chunks.pilot.jsonl

.venv/bin/python scripts/audit_chunk_quality.py \
  --chunks output/rich_chunks/rich_chunks.pilot.jsonl \
  --stats data/ml/reports/chunk_quality_pilot.json
```

**Gate fail →** править prompt / merge rules (`chunk_qa_prompt.py`, `merge_chunk_qa_fixes.py`), **не** тратить Cursor $.

---

## 5. Фаза A — команды (копипаст)

```bash
cd protocol   # корень репозитория

# A1
.venv/bin/python scripts/audit_chunk_quality.py \
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
  --stats data/ml/reports/chunk_quality_baseline.json \
  --report data/ml/reports/chunk_quality_baseline.md

# A2
.venv/bin/python scripts/build_chunk_qa_queue_tiered.py \
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
  --fixes data/ml/chunk_qa_fixes_merged.jsonl \
  --kz-folder clients_consult \
  --out data/ml/chunk_qa_queue_tiered.jsonl \
  --manifest data/ml/chunk_qa_queue_tiered_manifest.json

# A3
pytest tests/test_consult_retrieval.py \
  tests/test_patient_neurology_kz_case.py \
  tests/test_patient_dermatology_kz_case.py \
  tests/test_patient_phlebology_f1p_case.py \
  tests/test_patient_sanitization_boundary.py -q

# A4 (GOOGLE_API_KEY в env)
export CHUNK_QA_LLM=1 CHUNK_QA_LLM_BACKEND=gemini CHUNK_QA_MAX_OUT=16000
.venv/bin/python scripts/llm_chunk_qa.py \
  --queue data/ml/chunk_qa_queue_tiered.jsonl \
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
  --out data/ml/chunk_qa_fixes_pilot.jsonl \
  --limit 800 --batch-size 8
```

Полный план волн: [`ml/experiments/batch_r70_2026-06-28/GEMINI_FULL_QA_PLAN.md`](../ml/experiments/batch_r70_2026-06-28/GEMINI_FULL_QA_PLAN.md).

### D1–D2 — post-Wave A (копипаст)

```bash
cd protocol

# D1: свежий audit (не доверять отчёту до пустого merge)
.venv/bin/python scripts/audit_chunk_quality.py \
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
  --stats data/ml/reports/chunk_quality_post_deploy.json \
  --report data/ml/reports/chunk_quality_post_deploy.md \
  --baseline data/ml/reports/chunk_quality_baseline.json

# D2: RAG index после merge (обязательно, иначе JSONL на диске ≠ rerank)
.venv/bin/python scripts/build_chunk_embeddings.py

# Gate: локальный probe
.venv/bin/python scripts/run_symptom_icd_probe.py --local --no-gemini \
  --fixture tests/fixtures/symptom_icd_probe_100.jsonl \
  --out data/ml/reports/symptom_icd_probe_post_embed.jsonl
```

---

## 6. Cursor $ — только 2 Opus-сессии + Auto-код

### Когда какую модель

| Задача | Модель |
|--------|--------|
| Черновик, grep, правка 1 файла | **Auto** |
| Таблица root-cause по 5+ рубрикам | **Opus 1×** в конце |
| «Напиши 10 идей» без implement | ❌ не тратить Opus |
| Implement quick win после таблицы | **Auto Agent** |

---

### Прогон B1 — Routing (главный Cursor-$ ROI)

**Почему первый сейчас:** post-Wave A L1 **84.2%** без изменений - провалы **scoring vs routing vs chunks**. L1 tier **не читает chunk-RAG**; routing B1 двигает L2 и слабые `report_n_*`. Универсально для **всех** специальностей.

**Файлы (@):**  
`ml/experiments/batch_clients_consult_2026-06-28/ACTION_PLAN.md` ·  
`ml/experiments/batch_r70_2026-06-28/GEMINI_FULL_QA_PLAN.md` (§3.3) ·  
`clinical_knowledge/consult_retrieval.py` ·  
`clinical_knowledge/patient_protocol_filter.py` ·  
`tests/test_consult_retrieval.py`

**Промпт Agent (Auto, финал Opus только для таблицы):**

```text
Protocol: consult-review для КЗ ЛЮБОЙ специальности. Corpus: 478 PDF, 24 rubrics,
~59k chunks. Batch ~84% overall; failures mix routing (wrong protocol PDF, age path),
rag_chunks_n=0, and scoring/sparse rules — NOT chunk quality alone.

Задача (строго):
1. Классифицируй типы провалов: routing | chunks | scoring. По 1 примеру из
   РАЗНЫХ рубрик (gastro, kard, pediatr, lor, uro, onko, therapy…) — cite ACTION_PLAN.
2. Для каждого типа: файл(ы) в коде, минимальный fix, один pytest.
3. Решение: что fix **после** Wave A (routing/scoring) vs что уже закрыл corpus QA.
4. L2 smoke matrix: 5 KZ / 5 rubrics, pass = rag_chunks_n>0, no preamble in cite.

НЕ: переписать RAG, full repo review, specialty-specific hacks.

Output: markdown table (max 12 rows) + ONE recommended PR scope (≤3 files).
```

**Done когда:** ☐ merged PR + ☐ smoke matrix written + ☐ pytest green.

---

### Прогон B2 — B2C: implement 3 wins (не brainstorming)

**Ошибка прошлой версии чек-листа:** «10 улучшений Opus» без кода — сжигает $ без shippable result.

**Сначала** прочитать §17 gaps в `docs/architecture-b2c-patient.md`. **Потом** Agent:

**Файлы (@):**  
`docs/architecture-b2c-patient.md` (§17–18) ·  
`clinical_knowledge/patient_report_v2.py` ·  
`clinical_knowledge/patient_question_builder.py` ·  
`patient-ui.js` ·  
`tests/test_patient_report_v2_schema.py`

**Промпт Agent (Auto, implement mode):**

```text
B2C «Проверь КЗ» — для ЛЮБОЙ специальности. Read architecture-b2c-patient.md §17 gaps.

Implement EXACTLY 3 P0 items from §17 that are:
- universal (not one ICD/specialty)
- testable with existing or 1 new pytest
- ≤200 lines total diff

Do NOT: chunk QA, rag_server refactor, new payment provider.

After code: run pytest tests/test_patient_*case*.py tests/test_patient_report_v2_schema.py -q
List what you changed and why patients of any specialty benefit.
```

**Done когда:** ☐ 3 пункта в коде ☐ pytest green ☐ BUILD_VERSION bumped.

---

### Прогон B3 — Sanitization (15 мин, Auto)

**Файлы (@):** `tests/test_patient_sanitization_boundary.py` · `clinical_knowledge/patient_review.py` · `api/patient/openapi.yaml`

```text
Audit B2C API leak of B2B fields (send_gate, cisz, raw alignment). Max 5 issues.
If tests already cover — say "OK" in 3 lines. Else minimal fix + test. No rag_server wide refactor.
```

---

## 7. L2 smoke (после wave A + embeddings на Render)

| Fixture | Рубрика | Pass |
|---------|---------|------|
| gastro_1 | гастро | `rag_chunks_n>0` в **JSON ответа** |
| kard_1 | кардио | цитата без preamble |
| pediatr_1 | педиатрия | overall ≥ batch baseline |
| report_lor_1 | ЛОР | `rag_chunks_n>0` |
| report_urolog_1 | урология | `rag_chunks_n>0` |

```bash
# batch даёт overall%, но не всегда rag_chunks_n - смотреть report.json:
.venv/bin/python scripts/run_clients_consult_render_batch.py \
  --base https://protocol-bimy.onrender.com --tier L2 \
  --cases gastro_1,kard_1,pediatr_1,report_lor_1,report_urolog_1 \
  --out ml/experiments/batch_smoke_latest/l2_sample
# jq '.reports[] | {case_id, overall_pct, rag_chunks_n: .retrieval_top}' report.json
```

Не гонять full batch 29 KZ до green smoke 5 **и** merged B1 PR.

---

## 8. Антипаттерны (сжигают $70 и время)

| ❌ | ✅ вместо |
|----|-----------|
| Opus + весь repository | B1 routing table + 1 PR |
| 10 идей B2C без кода | B2 implement 3 из §17 |
| Chunk QA >800 до gate | Gate → wave A |
| `build_kz_weak_chunk_qa_queue` (legacy) | `build_chunk_qa_queue_tiered` |
| Промпт «под неврологию» | Матрица 24 rubrics |
| L2 batch 29 KZ до routing fix | Smoke 5 rubrics |
| Deploy corpus без embeddings | D2 `build_chunk_embeddings.py` |
| Merge из пустого `final.jsonl` | merge из `section_mapped.jsonl` |
| Wave B 5k без gate | retry + continuous queue only |
| Opus для pytest/fix typo | Auto |

---

## 9. Состояние Render (проверять перед wave)

```bash
ssh srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com \
  'wc -l /var/data/output/rich_chunks/rich_chunks.jsonl; ls -lh /var/data/output/rich_chunks/rich_chunks.jsonl'
curl -s https://protocol-bimy.onrender.com/api/version | python3 -m json.tool | head -8
```

Ожидаемо после заливки: `rich_chunks.jsonl` **~57k строк**, **не 0 байт**. В lazy/manifest режиме `/api/corpus-stats` может показывать `corpus_chunks: 0` - **проверять файл на диске по SSH**, не только API.

**Сегодняшние** изменения на диске: `find /var/data -type f -newermt 'YYYY-MM-DD'`.

---

## 10. Связанные документы

| Документ | Зачем |
|----------|--------|
| [`GEMINI_FULL_QA_PLAN.md`](../ml/experiments/batch_r70_2026-06-28/GEMINI_FULL_QA_PLAN.md) | chunk waves, gates |
| [`ACTION_PLAN.md`](../ml/experiments/batch_clients_consult_2026-06-28/ACTION_PLAN.md) | batch KZ failures |
| [`architecture-b2c-patient.md`](architecture-b2c-patient.md) | B2C gaps §17 |
| [`chunk-quality-pipeline.md`](chunk-quality-pipeline.md) | rule QA modules |

---

## Шпаргалка: оптимально ли тратить $70 так?

**Да** (после Wave A), если месяц выглядит так:

1. **$0** — audit 57k + embeddings + probe gate
2. **Cursor Auto + 1× Opus** — B1 routing (главный рычаг для слабых KZ)
3. **Cursor Auto** — B2 три B2C fix + B3 sanitization
4. **GOOGLE** — только retry/continuous, не blind Wave B
5. **$10+ остаток** — доработки после smoke

**Нет**, если $70 ушли на повторный bulk chunk QA, Opus без PR в `main`, или deploy corpus без embeddings.

---

*Версия чек-листа: 2026-06-29 (post-Wave A). Предыдущая логика A→B→C сохранена; добавлены §0, неделя D и уроки deploy.*
