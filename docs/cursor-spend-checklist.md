# Cursor $70: как тратить с максимальным ROI (Protocol)

> **Продукт:** проверка **любого** КЗ **любой** специальности против **478** протоколов Минздрава РБ (B2B врач + B2C пациент).  
> **Не путать:** десятки тысяч КЗ в год на рынке ≠ один batch из 30 fixtures — batch **измеряет** pipeline, chunk QA **улучшает** retrieval для всех.

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

| Приоритет | Рычаг | Кошелёк | Ожидаемый эффект |
|-----------|--------|---------|------------------|
| **1** | Tiered **chunk pilot → gate → wave A** | **GOOGLE_API_KEY** | ↑ recall, ↓ preamble в цитатах **по всем 24 рубрикам** |
| **2** | **Routing / retrieval** (универсальный код) | **$0 код + Cursor $** на диагностику | ↑ batch overall, `rag_chunks_n>0` на L2 |
| **3** | **3 quick wins B2C** в код | **Auto** (реализация) | ↑ конверсия «понял → пришёл с вопросами» |
| **4** | B2B→B2C sanitization | **Auto** | ↓ риск утечки send_gate/ЦИСЗ |

**Вывод:** $70 Cursor **оптимальны**, если **не дублируют п.1**, а тратятся на **п.2–4** (суждение + точечные PR). **Неоптимально:** Opus + «прочитай весь repo».

### North star (как понять, что деньги не зря)

| Метрика | Сейчас (ориентир) | Цель месяца |
|---------|-------------------|-------------|
| Chunk mean quality | ~0.936 (`audit`) | ≥0.938 после wave A |
| Batch KZ overall (L1) | ~81–84% (`clients_consult`) | нет регрессии; слабые рубрики ↑ |
| L2 smoke 5 рубрик | часть с `rag_chunks_n=0` | **все 5:** chunks>0, без preamble |
| B2C pytest matrix | neuro/derma/phleb pass | +1 новый fixture после fix |
| Cursor $70 остаток | — | ≥$10 в конце месяца (не сжечь в 1 день) |

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

### Неделя A — корпус (без Cursor $)

| ☐ | Действие | Команда / артеfact |
|---|----------|-------------------|
| A1 | Baseline audit | `scripts/audit_chunk_quality.py` → `data/ml/reports/chunk_quality_baseline.json` |
| A2 | Tier-очередь **24 рубрики** + ICD из всех KZ | `scripts/build_chunk_qa_queue_tiered.py` → manifest `by_rubric`, `kz_icd_protocol_paths` |
| A3 | Регрессия | `pytest tests/test_consult_retrieval.py tests/test_patient_*case*.py tests/test_patient_sanitization_boundary.py -q` |
| A4 | **Pilot 800** | `scripts/llm_chunk_qa.py --limit 800` (GOOGLE) |
| A5 | **Gate pilot** | merge + audit; см. §4 — **стоп**, если gate fail |

### Неделя B — Cursor $ (только после A5 или параллельно A3)

| ☐ | Прогон | Модель | ~$ | Deliverable |
|---|--------|--------|-----|-------------|
| B1 | **Routing root-cause** (все рубрики) | Auto → **1× Opus** | 20–25 | Таблица + 1 PR с тестом |
| B2 | **B2C: 3 quick wins** (не 10 идей) | Auto implement | 10–15 | 3 коммита + pytest |
| B3 | Sanitization boundary | Auto | 5–8 | ≤5 рисков или зелёный pytest |
| — | Резерв | Auto | 10+ | Фиксы после wave A |

### Неделя C — корпус на prod (GOOGLE, gate OK)

| ☐ | Wave A P0 + KZ-linked PDFs | `llm_chunk_qa.py --append` |
| ☐ | promote + upload Render | `scripts/upload_rich_chunks_render.sh … --gzip` |
| ☐ | L2 smoke **5 рубрик** | gastro, kard, pediatr, lor, uro fixtures |
| ☐ | Batch snapshot | `scripts/run_clients_consult_render_batch.py` (если есть) |

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
  --chunks output/rich_chunks/rich_chunks.final.jsonl \
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

**Почему первый:** batch r70 ~84%, но провалы **скoring vs routing vs chunks** — без классификации chunk wave может не поднять overall. Универсально для **всех** специальностей.

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
3. Решение: что MUST fix до wave A chunk deploy vs что wave A alone fixes.
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

## 7. L2 smoke (после wave A на Render)

| Fixture | Рубрика | Pass |
|---------|---------|------|
| gastro_1 | гастро | `rag_chunks_n>0` |
| kard_1 | кардио | цитата без preamble |
| pediatr_1 | педиатрия | overall ≥ batch baseline |
| report_lor_1 | ЛОР | |
| report_urolog_1 | урология | |

Не гонять full batch 29 KZ до green smoke 5.

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
| Opus для pytest/fix typo | Auto |

---

## 9. Состояние Render (проверять перед wave)

```bash
ssh srv-d78he6h5pdvs73b1kufg@ssh.oregon.render.com \
  'ls -lh /var/data/output/rich_chunks/rich_chunks.jsonl; ls -la /var/data/chunk_qa/ | head'
curl -s https://protocol-bimy.onrender.com/api/version | python3 -m json.tool | head -5
```

Ожидаемо после заливки: `rich_chunks.jsonl` + каталог `chunk_qa/` (pilot queue/fixes). **Сегодняшние** изменения — `find /var/data -type f -newermt 'YYYY-MM-DD'`.

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

**Да**, если месяц выглядит так:

1. **GOOGLE** — pilot 800 + gate + (maybe) wave A → **↑ качество для всех специальностей**
2. **Cursor Opus ×2** — routing diagnosis + (optional) synthesis, не ideation
3. **Cursor Auto** — 3 B2C shippable fixes + sanitization
4. **$10+ остаток** — доработки после smoke

**Нет**, если $70 ушли на «улучши всё» в чате без merge в `main` и без движения north star метрик.
