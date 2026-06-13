# План безопасной интеграции Protocol Summary Cards

Документ описывает поэтапное добавление слоя **Protocol Summary Cards** без переписывания текущего legacy-пайплайна. Источник требований: `docs/cursor_task_protocol_summary_cards_with_legacy_fallback.md`.

---

## 1. Текущий baseline (не трогаем поведение)

Legacy-контур остаётся **default** и **fallback**:

```text
PDF → chunks.jsonl + protocol_cards.jsonl
  → catalog/rules (corpus + path + enrichment)
  → match_protocol_cards → run_rule_checker
  → build_compliance_report → consult_report
```

Ключевые модули (расширяем, не заменяем):

| Модуль | Роль |
|--------|------|
| `clinical_knowledge/loader.py` | Загрузка cards, conditions, rules |
| `clinical_knowledge/rule_checker.py` | Детерминированная проверка |
| `clinical_knowledge/compliance_engine.py` | Итоговый score, evidence map |
| `clinical_knowledge/consult_analysis.py` | Оркестратор CLI/API |
| `consult_review_pipeline.py` | Online RAG + structured branch |
| `corpus_pipeline/protocol_cards.py` | Legacy registry |

**Контракт:** при `PROTOCOL_SUMMARY_MODE=legacy` (или `PROTOCOL_SUMMARY_ENABLED=0`) байт-в-байт тот же путь, что до интеграции.

---

## 2. Новый слой (additive)

```text
clinical_knowledge/protocol_summary/
  schema.py           # Pydantic ProtocolSummary
  config.py           # env-переменные
  validator.py        # strict + validation_reports
  loader.py           # YAML/JSON → summaries, rules lookup
  builder.py          # documents/chunks → draft YAML
  summary_to_rules.py # Summary → ProtocolRule (+ rule_source=summary)
  summary_to_rag.py   # Summary → summary_chunks
  method_selector.py  # legacy | summary | hybrid
  summary_compare.py  # сравнение режимов на одном КЗ
  summary_quality.py  # корпусный отчёт покрытия

data/protocol_summaries/
  yaml/ json/ drafts/ reviewed/ validation_reports/ comparison_reports/
```

---

## 3. Режимы работы

| Режим | Поведение |
|-------|-----------|
| `legacy` | Только текущий pipeline (default) |
| `summary` | Только rules из summary; при invalid/missing → fallback или `insufficient_protocol_data` |
| `hybrid` | Summary primary; legacy fallback + evidence; LLM/RAG не повышает score |

Env (см. `protocol_summary/config.py`):

- `PROTOCOL_SUMMARY_MODE=hybrid` (default **legacy** до стабилизации QA)
- `PROTOCOL_SUMMARY_ENABLED=1`
- `PROTOCOL_SUMMARY_FALLBACK_TO_LEGACY=1`
- `PROTOCOL_SUMMARY_MIN_REVIEW_STATUS=draft`

**Безопасный default на этапе внедрения:** `MODE=legacy`, `ENABLED=0` - новый код загружается, но не влияет на production до явного включения.

---

## 4. Точки интеграции (минимальный diff)

### 4.1 `rule_model.ProtocolRule`

Добавить поля (с дефолтами - backward compatible):

```python
rule_source: Literal["legacy", "summary", "manual", "table", "llm_draft"] = "legacy"
generated_from_summary: bool = False
summary_id: str | None = None
summary_version: str | None = None
condition_id: str | None = None
```

`legacy_rule_to_protocol_rule()` всегда ставит `rule_source="legacy"`.

### 4.2 `consult_schema`

Расширить `EvidenceMapItem.rule_source`, `ComplianceReport`:

```python
analysis_mode: Literal["legacy", "summary", "hybrid"] = "legacy"
protocol_summary_used: bool = False
fallback_to_legacy: bool = False
method_comparison: dict | None = None
```

Старые JSON-отчёты без этих полей - валидны (`extra="ignore"` на потребителе не нужен - дефолты).

### 4.3 `consult_analysis.analyze_consultation_text`

Новый параметр `analysis_mode: str | None = None` (из env если None):

1. `method_selector.resolve_analysis_plan(...)` → `{primary, fallback, compare}`
2. Legacy path - **без изменений** внутри `_run_legacy_analysis()`
3. Summary/hybrid - подмешать `load_summary_rules()` перед `run_rule_checker`
4. Hybrid merge в `build_compliance_report(..., summary_meta=...)`

### 4.4 `compliance_engine.build_compliance_report`

Опциональные kwargs:

- `analysis_mode`
- `summary_meta` (used cards, fallback flags)
- `legacy_rules_check` / `summary_rules_check` для hybrid compare

Конфликт summary vs legacy → `ComplianceIssue(manual_review)`.

### 4.5 `consult_report`

Секции в MD/HTML: режим, summary status, fallback, delta vs legacy.

### 4.6 RAG (`summary_to_rag` + опционально `rag_server`)

Phase 1: генерировать `data/protocol_summaries/summary_chunks.jsonl` offline.  
Phase 2: `retrieve(..., corpus="summary"|"hybrid")` - отдельный PR, не блокирует compliance.

---

## 5. Fallback strategy

```text
if mode == legacy OR not ENABLED:
    legacy only

if mode == summary:
    if valid summary for matched protocol_ids:
        summary rules
    elif FALLBACK_TO_LEGACY:
        legacy + flag fallback_to_legacy=true
    else:
        overall_status = insufficient_protocol_data

if mode == hybrid:
    run summary if valid
    always run legacy (cached) for evidence/fallback
    merge: summary rules win on conflict if review_status >= MIN_REVIEW_STATUS
    unfilled sections → legacy rules
    if summary invalid → legacy only + fallback flag
```

**Red flags:** safety_checker не отключается; summary red_flag_rule + legacy safety - union, не intersection.

**LLM:** `llm_score_ignored=True` сохраняется; summary не меняет deterministic score через LLM.

---

## 6. Риски и митигация

| Риск | Митигация |
|------|-----------|
| Слом regression-тестов KZ | Default `legacy`; после каждого этапа - `pytest tests/test_regression_kz_compliance.py` |
| Дублирование rules → двойной штраф | Hybrid dedupe по `rule_type + expected_items`; summary priority |
| Неполные draft YAML | Validator → `needs_human_review`; hybrid fallback |
| Производительность loader | `@lru_cache` как в `loader.py` |
| Schema drift | Pydantic + unit tests + validation_reports |
| Отсутствие summary для 478 PDF | Не блокирует legacy; quality report показывает % покрытия |

---

## 7. Этапы реализации (порядок)

| # | Этап | Артеfact | Проверка legacy |
|---|------|----------|-----------------|
| 1 | schema | `protocol_summary/schema.py`, docs | import-only, no pipeline change |
| 2 | validator | `validator.py`, reports | no pipeline change |
| 3 | loader | `loader.py` | no pipeline change |
| 4 | builder | `builder.py`, sample YAML | no pipeline change |
| 5 | summary_to_rules | `summary_to_rules.py`, `rule_source` | legacy rules unchanged |
| 6 | summary_to_rag | `summary_to_rag.py` | RAG optional |
| 7 | method_selector | `method_selector.py`, `config.py` | default legacy |
| 8 | compliance | `consult_analysis`, `compliance_engine` | `MODE=legacy` identical |
| 9 | reports | `consult_report`, compare, quality | legacy report fields preserved |
| 10 | tests | unit + regression modes | all legacy tests green |

---

## 8. CLI (этап 9-10)

```bash
python -m scripts.build_protocol_summaries [--limit N]
python -m scripts.validate_protocol_summaries
python -m scripts.export_protocol_summary_rules
python -m scripts.export_protocol_summary_rag
python -m scripts.compare_summary_vs_legacy --file kz.txt
python -m scripts.analyze_consultation --file kz.txt  # + --mode legacy|summary|hybrid
```

---

## 9. Критерии готовности этапа

- [ ] Legacy pytest suite green (`test_regression_kz_compliance`, `test_compliance_engine`, `test_consult_analysis`)
- [ ] Новые unit-тесты этапа green
- [ ] `PROTOCOL_SUMMARY_ENABLED=0` → нулевой diff в compliance output vs до PR
- [ ] Документация schema/workflow обновлена при изменении формата

---

## 10. Первая цель покрытия (builder)

1. Gastro MVP protocols (из `data/gastro_mvp/`)
2. Fixtures-linked conditions: K30, I80.1, L30, J06.8
3. Draft YAML в `data/protocol_summaries/drafts/` + validated copies в `yaml/`

Полный каталог 478 PDF - итеративно, без блокировки legacy.
