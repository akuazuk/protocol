# LLM-судья action-очереди МО: этапы A/B (v1)

Дата: 2026-08-05  
Статус: active  
Связанные:

- `2026-08-05-mo-eval-smirnova-concordance-v1.md` - детерминированные shadow findings;
- `2026-08-05-mo-case-protocol-suggest-v1.md` - подбор КП (отдельный модуль);
- `scripts/grade_kz_llm.py` - двухтировый грейдер L1-месяца (не путать с этим batch).

---

## 1. Контекст

Нужен **узкий** Gemini-прогон не на все ~500 МО дня, а только на
**action-очередь «Вчера»** (обычно ≤20 кейсов: P0/P1 + shadow).

Два этапа на кейс:

1. **A - диагноз:** возраст, жалобы, анамнез, обследования ↔ правильность диагноза.
2. **B - план:** рекомендации по обследованию и лечению ↔ адекватность плану на базе A.

Результат - shadow JSON для методиста. **Не** меняет primary `overall_pct` / warehouse
score, пока нет явного флага `MO_LLM_ACTION_JUDGE_PRIMARY=1` (в v1 выключен).

---

## 2. Метрики

| Метрика | Было | Цель v1 |
|--|--|--|
| Охват batch | нет | только `action_cases.items` дня |
| Время на очередь ~10 кейсов (3.6-flash, 2 этапа) | н/д | < 15 мин wall |
| Стоимость на очередь ~10 кейсов | н/д | < $1 на 3.6-flash |
| Валидный JSON A и B | н/д | 100% parse после retry×1 |
| Primary overall меняется | - | нет (shadow only) |

---

## 3. JSON-контракт

Общие правила:

- ответ модели - **один JSON-объект**, без markdown;
- `score_pct` всегда 0-100;
- `confidence` 0-1;
- `severity`: `P0` | `P1` | `P2` | `P3` | `none`;
- цитаты `evidence` - короткие фрагменты из КЗ (≤200 символов), не выдумывать;
- если данных нет: `unknown`, не фантазировать возраст/обследования.

### 3.1 Этап A - диагноз (`stage: "A"`)

```json
{
  "schema_version": 1,
  "stage": "A",
  "engine": "mo_llm_action_judge_v1",
  "case_id": "3646270",
  "visit_id": "3646270",
  "mis_id": "898517",
  "patient": {
    "age_years": 9,
    "audience": "pediatric",
    "age_source": "visit_meta|text|unknown"
  },
  "inputs_used": {
    "complaints": true,
    "anamnesis": true,
    "exams": false,
    "objective_status": true,
    "diagnosis": true
  },
  "diagnosis_assessment": {
    "score_pct": 35,
    "verdict": "poor",
    "summary_ru": "Диагноз M60 не закрывает отёк колена и хроническую хромоту.",
    "supported_by": ["болезненность прямой мышцы бедра"],
    "not_supported_by": ["отёк правого колена", "хромота 3 месяца"],
    "icd": {
      "code": "M60",
      "text": "Миозит …",
      "fit": "weak|adequate|strong|unknown"
    }
  },
  "chain": [
    {
      "from": "complaints",
      "to": "diagnosis",
      "outcome": "gap",
      "note": "Хромота/отёк сустава не отражены в Dx"
    },
    {
      "from": "exams",
      "to": "diagnosis",
      "outcome": "unknown",
      "note": "Обследований в КЗ нет"
    }
  ],
  "findings": [
    {
      "code": "finding_not_in_diagnosis",
      "severity": "P1",
      "text_ru": "Отёк колена есть в статусе, в диагнозе отсутствует",
      "evidence": "отёк правого коленного сустава"
    }
  ],
  "conclusion_ru": "Диагноз слабо согласован с клиникой; нужен DDx суставной патологии.",
  "confidence": 0.78,
  "needs_human": true
}
```

Допустимые `verdict`: `good` | `acceptable` | `review` | `poor` | `critical`.  
Допустимые `outcome`: `ok` | `gap` | `contradiction` | `unknown`.

### 3.2 Этап B - план (`stage: "B"`)

На вход модели обязательно передаётся **сжатый итог A** (`stage_a_digest`), не весь сырой A.

```json
{
  "schema_version": 1,
  "stage": "B",
  "engine": "mo_llm_action_judge_v1",
  "case_id": "3646270",
  "visit_id": "3646270",
  "mis_id": "898517",
  "stage_a_ref": {
    "diagnosis_score_pct": 35,
    "diagnosis_verdict": "poor",
    "key_gaps": ["finding_not_in_diagnosis", "chronic_pediatric_limp"]
  },
  "plan_assessment": {
    "exam_recommendations": {
      "score_pct": 20,
      "verdict": "poor",
      "present": [],
      "missing_suggested": ["УЗИ коленного сустава", "ОАК/СОЭ/СРБ"],
      "summary_ru": "При хромоте ≥4 нед и отёке сустава обследований нет."
    },
    "treatment_recommendations": {
      "score_pct": 40,
      "verdict": "review",
      "present": ["ибупрофен", "массаж обоих бёдер"],
      "concerns": ["laterality_mismatch", "underworkup_before_therapy"],
      "summary_ru": "Симптоматическое лечение без дообследования; массаж билатеральный."
    },
    "follow_up": {
      "score_pct": 15,
      "verdict": "poor",
      "kind": "on_worsening_only|scheduled|unclear|absent",
      "summary_ru": "Контроль только при ухудшении - недостаточно."
    },
    "score_pct": 25,
    "verdict": "poor"
  },
  "findings": [
    {
      "code": "underworkup_chronic_red_flag",
      "severity": "P1",
      "text_ru": "Нет imaging/labs при хронической педиатрической хромоте",
      "evidence": "контроль при отрицательной динамике"
    }
  ],
  "conclusion_ru": "План не закрывает пробелы диагноза; сначала дообследование, затем терапия.",
  "confidence": 0.74,
  "needs_human": true
}
```

### 3.3 Обёртка batch-строки (файл результата)

Каждая строка JSONL:

```json
{
  "date": "2026-08-04",
  "case_id": "3646270",
  "visit_id": "3646270",
  "mis_id": "898517",
  "queue_reason": "P0: Критический дефект по №55",
  "queue_severity": "P0",
  "model_a": "gemini-3.6-flash",
  "model_b": "gemini-3.6-flash",
  "latency_ms_a": 12400,
  "latency_ms_b": 15100,
  "stage_a": { "...": "контракт A" },
  "stage_b": { "...": "контракт B" },
  "error": null
}
```

Путь по умолчанию (не в git):  
`data/medical_exams/llm_action_judge/YYYY/MM/DD/judges.jsonl`  
на Render: `/var/data/medical_exams/llm_action_judge/...`

---

## 4. Команда batch (только action-очередь)

Dry-run (без Gemini, только список очереди + валидация фикстуры):

```bash
python3 scripts/run_mo_action_queue_llm_judge.py \
  --date 2026-08-04 \
  --source render \
  --dry-run
```

Боевой прогон A+B на очереди дня (ключ Gemini должен быть доступен; с Mac часто geo-block → лучше Render shell / машина с доступом):

```bash
python3 scripts/run_mo_action_queue_llm_judge.py \
  --date 2026-08-04 \
  --source render \
  --stages ab \
  --model gemini-3.6-flash \
  --concurrency 3 \
  --limit 20 \
  --out data/medical_exams/llm_action_judge/2026/08/04/judges.jsonl
```

Только этап A:

```bash
python3 scripts/run_mo_action_queue_llm_judge.py --date yesterday --stages a --source render
```

Локальный отчёт (если уже скачан publish):

```bash
python3 scripts/run_mo_action_queue_llm_judge.py \
  --date 2026-08-04 \
  --source local \
  --medical-exams-root data/medical_exams \
  --dry-run
```

---

## 5. Шаги реализации

- [x] Зафиксировать JSON-контракт A/B и batch-команду в этом плане.
- [x] Чистый модуль validate/prompt: `clinical_knowledge/mo_llm_action_judge.py`.
- [x] CLI `scripts/run_mo_action_queue_llm_judge.py` (dry-run + execute).
- [x] Unit-тесты на parse/validate.
- [ ] Пилотный прогон на очереди одного дня (Render), отчёт методисту.
- [ ] UI: блок «LLM-судья A/B» в case detail (только чтение JSONL/API) - отдельная итерация.

---

## 6. Риски

| Риск | Митигация |
|--|--|
| ПДн в ответах LLM | писать только под `medical_exams/`; не в git |
| Geo-block с Mac | batch с Render / NL VPN |
| Путают с primary score | `engine` id + shadow-only |
| Раздувание промпта PDF КП | v1 без полного PDF; только слоты КЗ + digest A |
| Двойной смысл с `grade_kz_llm` | другой CLI, только action queue |

---

## 7. Definition of Done v1 (docs+CLI)

1. Контракт A/B описан и валидируется тестами.
2. `--dry-run` печатает case_id из action-очереди выбранного дня.
3. `--stages ab` готов к пилоту без записи в primary warehouse.

---

## 8. Первая безопасная команда

```bash
cd /private/tmp/protocol-task-mo-llm-action-queue-judge-pc1
python3 scripts/run_mo_action_queue_llm_judge.py --date 2026-08-04 --source render --dry-run
```
