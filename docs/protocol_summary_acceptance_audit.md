# Protocol Summary Cards — отчёт приёмочного аудита

**Дата:** 2026-05-31  
**ТЗ:** `docs/cursor_task_protocol_summary_cards_with_legacy_fallback.md`  
**Коммит:** `5a218ad` (main)  
**Метод:** статический обзор кода + запуск CLI/тестов без изменений в репозитории.

---

## Сводка

| Область | Статус |
|---------|--------|
| Инфраструктура (schema, builder, validator, loader, export) | **В основном выполнено** |
| Корпус карточек (478 PDF → 475 drafts, 465 yaml/json) | **Частично** (качество draft, 0 approved) |
| Режимы legacy/summary/hybrid | **Частично** (код есть, online-путь не активирует summary) |
| Summary → rules / RAG | **Частично** (генерация есть, RAG retrieve не подключён) |
| Online pipeline + отчёты + comparison | **Частично** |
| Regression на КЗ (3 режима) | **Не выполнено** (режимы дают одинаковый legacy-результат) |
| Продакшн-ready summary/hybrid | **Нет** — оставить `legacy` |

---

## 1. Режимы работы

| Вопрос | Статус | Детали |
|--------|--------|--------|
| Есть ли режим **legacy**? | **Выполнено** | `clinical_knowledge/protocol_summary/method_selector.py` → `resolve_analysis_plan()` при `enabled=False` или `mode=legacy` возвращает `use_legacy=True`, `use_summary=False`. |
| Есть ли режим **summary**? | **Выполнено** | `method_selector.py:66–101` — primary_source=`summary`, `use_summary=True`. |
| Есть ли режим **hybrid**? | **Выполнено** | `method_selector.py:103–127` — summary primary + legacy для fallback/evidence. |
| Где реализован выбор режима? | **Выполнено** | `config.py` (`ProtocolSummaryConfig.from_env()`), `method_selector.resolve_analysis_plan()`, вызов из `consult_analysis.analyze_consultation_text()`, CLI `scripts/analyze_consultation.py --mode`. |
| Что если summary-карточка **отсутствует**? | **Частично** | `method_selector`: summary → fallback legacy (`notes: "summary missing/invalid → fallback to legacy"`) при `PROTOCOL_SUMMARY_FALLBACK_TO_LEGACY=1`. **Но:** ICD-fallback в `consult_analysis.py:148–159` срабатывает только если `plan.use_summary=True`; при hybrid без matched protocol_id `use_summary=False` → ICD-поиск не выполняется. |
| Что если summary **invalid**? | **Выполнено** | `validator.summary_is_usable()` → `False` при `status=invalid`; карточка не попадает в `plan.usable_summaries`. При `FALLBACK_TO_LEGACY=1` — legacy; иначе `insufficient_protocol_data`. |
| Работает ли **fallback на legacy**? | **Частично** | Логика в `method_selector` есть; на практике при типичном КЗ без совпадения `protocol_id` карточек и матчей summary-ветка не активируется (см. §19). |
| Режим **по умолчанию**? | **Выполнено** | `PROTOCOL_SUMMARY_ENABLED=0`, `PROTOCOL_SUMMARY_MODE=legacy` (`config.py:49–50`). Поведение идентично legacy baseline. |

**Команда проверки:**
```bash
.venv/bin/python -c "
from clinical_knowledge.protocol_summary.method_selector import resolve_analysis_plan
print(resolve_analysis_plan(mode='legacy'))
print(resolve_analysis_plan(mode='hybrid', matched_protocol_ids=['test_gastro_k30'], enabled=True))
"
```

---

## 2. Env-переменные

Все переменные читаются в `clinical_knowledge/protocol_summary/config.py` → `ProtocolSummaryConfig.from_env()`.

| Переменная | Где читается | Default | Влияние |
|------------|--------------|---------|---------|
| `PROTOCOL_SUMMARY_MODE` | `config._env_mode()` | `legacy` | `legacy` / `summary` / `hybrid` → `method_selector.resolve_analysis_plan()`. |
| `PROTOCOL_SUMMARY_ENABLED` | `config._env_bool(..., False)` | `0` (false) | `False` → всегда legacy-план, summary-модули не участвуют в online-пути. |
| `PROTOCOL_SUMMARY_STRICT_VALIDATION` | `config._env_bool(..., True)` | `1` (true) | `validator.validate_protocol_summary()`: strict требует `quote` в `source_ref`. |
| `PROTOCOL_SUMMARY_FALLBACK_TO_LEGACY` | `config._env_bool(..., True)` | `1` (true) | summary без valid card → legacy вместо `insufficient_protocol_data`. |
| `PROTOCOL_SUMMARY_COMPARE_WITH_LEGACY` | `config._env_bool(..., True)` | `1` (true) | hybrid-план: флаг `compare_with_legacy` (но сравнение в pipeline **не вызывается**). |
| `PROTOCOL_SUMMARY_GENERATE_RULES` | `config._env_bool(..., True)` | `1` (true) | Зарезервировано; builder всегда может экспортировать rules через scripts. |
| `PROTOCOL_SUMMARY_GENERATE_RAG` | `config._env_bool(..., True)` | `1` (true) | `builder.build_protocol_summaries()` → `write_summary_rag_jsonl()` если не `--no-rag`. |
| `PROTOCOL_SUMMARY_MIN_REVIEW_STATUS` | `config._env_review_min()` | `draft` | `validator.review_status_acceptable()` — порог `draft`/`reviewed`/`approved`. |

Дополнительно: `PROTOCOL_SUMMARY_DATA_ROOT` (default `data/protocol_summaries`).

CLI `--mode` в `analyze_consultation.py:50–53` принудительно выставляет `ENABLED` и `MODE` перед анализом.

---

## 3. Protocol Summary schema

Файл: `clinical_knowledge/protocol_summary/schema.py` (Pydantic).

| Модель | Статус | Основные поля |
|--------|--------|---------------|
| `ProtocolSummary` | **Выполнено** | `protocol_id`, `summary_version`, `extraction_status`, `review_status`, `source`, `rubric`, `applicability`, `conditions[]`, `global_red_flags`, `validation` |
| `ProtocolSource` | **Выполнено** | `title`, `url`, `local_path`, `document_sha256`, `approval_date/number`, `document_year` |
| `ProtocolRubric` | **Выполнено** | `name`, `slug`, `specialty_slugs[]` |
| `ProtocolApplicability` | **Выполнено** | `population[]`, `age_min/max_years`, `sex`, `pregnancy`, `care_setting[]` |
| `ConditionSummary` | **Выполнено** | `condition_id`, `name`, `icd10_codes[]`, criteria, exams, treatment, follow_up, red_flags, `kz_checklist` |
| `SourceRef` | **Частично** | В ТЗ отдельная модель; реализована как `SummarySourceRef` + алиас через `consult_schema.SourceRef` в `to_source_ref()` |
| `DiagnosisStructure` | **Выполнено** | `required_components[]`, `optional_components[]`, `examples[]`, `source_refs[]` |
| `CriteriaBlock` | **Выполнено** | `required[]`, `optional[]`, `exclusion[]` → `CriterionItem` |
| `ExamRequirement` | **Выполнено** | `name`, `requirement_level`, `exam_type`, `source_ref`, `required_if`, `timing` |
| `TreatmentBlock` | **Выполнено** | `drugs[]`, `drug_groups[]`, `non_drug[]`, `procedures[]`, `surgery[]` |
| `DrugTreatmentItem` | **Выполнено** | `drug_name`, `dose_text`, `frequency_text`, `duration_text`, `source_ref` |
| `FollowUpRequirement` | **Выполнено** | `text`, `timing`, `expected_actions[]`, `source_ref` |
| `RedFlagItem` | **Выполнено** | `text`, `red_flag_type`, `severity`, `expected_actions[]`, `cap_if_unhandled`, `source_ref` |
| `KzChecklist` | **Выполнено** | `must_have[]`, `should_have[]`, `conditional[]`, `warnings[]` |

Тесты: `tests/test_protocol_summary_schema.py` (3 passed).

---

## 4. YAML/JSON-хранение

| Папка | Статус | Факт |
|-------|--------|------|
| `data/protocol_summaries/yaml/` | **Выполнено** | ~470 файлов |
| `data/protocol_summaries/json/` | **Выполнено** | ~469 файлов |
| `data/protocol_summaries/drafts/` | **Выполнено** | 478 файлов |
| `data/protocol_summaries/reviewed/` | **Не выполнено** | Каталог отсутствует; loader ищет его первым (`loader._summary_search_dirs`) |
| `data/protocol_summaries/validation_reports/` | **Выполнено** | 475 `.md` |
| `data/protocol_summaries/comparison_reports/` | **Не выполнено** | Каталог отсутствует; сравнение пишет в `data/reports/method_comparison/` |

### Пример YAML (тестовая карточка)

`data/protocol_summaries/yaml/test_gastro_k30.yaml`:

```yaml
protocol_id: test_gastro_k30
source:
  title: "Тест — функциональная диспепсия"
  local_path: "minzdrav_protocols/gastroenterologiya/test.pdf"
conditions:
  - condition_id: k30_functional_dyspepsia
    name: "Функциональная диспепсия"
    icd10_codes: [K30]
    required_exams:
      - name: "ЭГДС"
        requirement_level: required
        source_ref:
          page_start: 14
          section_title: "Диагностика"
          quote: "ЭГДС показана при диспепсических жалобах"
    red_flags:
      - text: "опухолевое образование"
        red_flag_type: possible_malignancy
        severity: critical
```

### Пример JSON (корпус)

`data/protocol_summaries/json/gastroenterologiya_кп_диагностика_лечение_пациентов_вз_нас_заболеваниями_желчнog.json` — зеркало YAML с полями `protocol_id`, `source`, `rubric`, `conditions[]` (множественные нозологии на один PDF).

---

## 5. Builder

| Пункт | Статус | Детали |
|-------|--------|--------|
| Модуль | **Выполнено** | `clinical_knowledge/protocol_summary/builder.py` |
| Команда | **Выполнено** | `python -m scripts.build_protocol_summaries [--limit N] [--rubric SLUG] [--no-publish] [--no-rag]` |
| Вход | **Выполнено** | `output/registry/protocol_cards.jsonl`, `output/chunks/chunks.jsonl` |
| Выход | **Выполнено** | `drafts/`, `yaml/` (valid), `json/`, `validation_reports/`, `summary_chunks.jsonl`, `summary_quality_report.md` |
| Извлекаемые поля | **Частично** | ICD из cards/chunks, exams (regex + procedures), drugs (chunk drugs + dose regex), red_flags (regex), follow_up, title из filename, rubric из card |
| Заглушки | **Частично** | `clinical_criteria`/`diagnostic_criteria`/`diagnosis_structure` часто пусты; exams часто `requirement_level=recommended` вместо `required`; `kz_checklist` пуст; `contraindications[]` пуст |
| Несколько нозологий в PDF | **Выполнено** | `_conditions_from_cards_and_chunks()` — до 24 conditions по ICD на PDF (`builder.py:153–161`) |

---

## 6. Validator

Файл: `clinical_knowledge/protocol_summary/validator.py`.

| Проверка | Статус |
|----------|--------|
| `protocol_id` | **Выполнено** (`missing_protocol_id`) |
| `source.title` | **Выполнено** |
| `source.url` или `local_path` | **Выполнено** |
| `rubric.name` | **Выполнено** |
| `applicability.population` | **Частично** — warning если пуст, не error |
| наличие `conditions` | **Выполнено** |
| `condition_id`, `condition.name` | **Выполнено** |
| `icd10_codes` или причина | **Частично** — warning `missing_icd10`, не error |
| `source_ref` у clinical items | **Выполнено** |
| `page_start` или `section_title` | **Выполнено** (`incomplete_source_ref`) |
| дубли conditions | **Выполнено** |
| дубли exam requirements | **Выполнено** (warning) |
| required/conditional не смешаны | **Выполнено** (warnings `required_exam_level`, `conditional_exam_level`) |
| age vs population конфликт | **Выполнено** (`age_population_conflict`) |
| critical red flag без `expected_actions` | **Выполнено** (error) |

### Пример validation report

`data/protocol_summaries/validation_reports/gastroenterologiya_кп_диагностика_лечение_пациентов_вз_нас_заболеваниями_желчнog.md`:

```markdown
# Validation: gastroenterologiya_кп_диагностика_лечение_...

- **status:** needs_human_review
- **review_status:** not_reviewed
- **extraction_status:** auto_extracted

## Warnings
- `required_exam_level` conditions[k82_8_...].required_exams[0]: required_exams содержит элемент не с level=required
- `duplicate_exam` conditions[k83_4_...].required_exams[0]: Дублирующееся обследование
```

Команда: `python -m scripts.validate_protocol_summaries` — отрабатывает на 475 файлах (~26 с).

---

## 7. Summary → rules

Файл: `clinical_knowledge/protocol_summary/summary_to_rules.py`.

| rule_type | Статус |
|-----------|--------|
| `diagnosis_structure_rule` | **Выполнено** |
| `clinical_criterion_rule` | **Выполнено** |
| `diagnostic_criterion_rule` | **Выполнено** |
| `required_exam_rule` | **Выполнено** |
| `conditional_exam_rule` | **Выполнено** |
| `treatment_group_rule` | **Выполнено** |
| `drug_rule` / `drug_dose_rule` / `drug_duration_rule` | **Частично** — один drug → один тип (dose/duration перекрывают друг друга) |
| `non_drug_rule` | **Выполнено** |
| `follow_up_rule` | **Выполнено** |
| `routing_rule` | **Выполнено** |
| `red_flag_rule` | **Выполнено** |
| `contraindication_rule` | **Не выполнено** — тип в `_LEGACY_TYPE_MAP`, генерация из `cond.contraindications` отсутствует |
| age/sex/pregnancy applicability rules | **Не выполнено** — applicability только в `RuleApplicability`, отдельные `age_applicability_rule` не создаются |

### Пример сгенерированного правила

Из `test_gastro_k30` (red flag):

```json
{
  "rule_id": "test_gastro_k30__k30_functional_dyspepsia__red_flag_0",
  "rule_type": "red_flag_rule",
  "rule_source": "summary",
  "generated_from_summary": true,
  "condition_id": "k30_functional_dyspepsia",
  "keywords": ["опухолевое образование", "нельзя исключить инвазию"],
  "red_flag_type": "possible_malignancy",
  "source": { "protocol_id": "test_gastro_k30", "quote": "...", "page": 30 }
}
```

Поля `protocol_id`, `icd10_codes`, `severity`, `evidence_targets` — в `ProtocolRule` (`summary_to_rules._rule_base()`).

Тесты: `tests/test_summary_to_rules.py` (3 passed).

---

## 8. Summary → RAG

| Пункт | Статус |
|-------|--------|
| Типы чанков (9 видов) | **Выполнено** | `summary_to_rag.SUMMARY_CHUNK_TYPES` + `condition_to_summary_chunks()` |
| Файл корпуса | **Выполнено** | `data/protocol_summaries/summary_chunks.jsonl` — 10 648 строк |
| RAG ищет raw + summary одновременно | **Не выполнено** | В `rag_server.py` нет упоминаний `summary_chunks` / `summary_overview` |
| Hybrid boost summary_chunks | **Не выполнено** | Не реализовано |

### Пример summary chunk

```json
{
  "chunk_id": "...__summary_overview",
  "protocol_id": "akusherstvo_ginekologiya_...",
  "condition_id": "i80_...",
  "section_type": "summary_overview",
  "generated_from_summary": true,
  "icd10_codes": ["I80"],
  "text": "Протокол: КП Диагностика... МКБ-10: I80."
}
```

Команда: `python -m scripts.export_protocol_summary_rag` → `Wrote .../summary_chunks.jsonl`.

---

## 9. Loader

Файл: `clinical_knowledge/protocol_summary/loader.py`.

| Функция | Статус | Пример вызова |
|---------|--------|---------------|
| `load_protocol_summaries()` | **Выполнено** | `load_protocol_summaries(usable_only=True)` |
| `load_summary_by_protocol_id()` | **Выполнено** | `load_summary_by_protocol_id("test_gastro_k30")` |
| `find_conditions_by_icd()` | **Выполнено** | `find_conditions_by_icd("K30")` → 1 condition |
| `find_conditions_by_text()` | **Выполнено** | `find_conditions_by_text("диспепсия", limit=5)` |
| `load_summary_rules()` | **Выполнено** | `load_summary_rules(usable_only=True)` → tuple[ProtocolRule] |

Тесты: `tests/test_summary_loader.py` (3 passed).

---

## 10. Online pipeline

| Шаг ТЗ | Статус | Файл / функция |
|--------|--------|----------------|
| Парсинг КЗ | **Выполнено** | `consult_parser.parse_consultation()` |
| Диагнозы, МКБ, демография | **Выполнено** | `consult_analysis.facts_from_document()` |
| Поиск Protocol Summary conditions | **Частично** | `loader.find_conditions_by_icd()` — только если `plan.use_summary=True` |
| Applicability | **Выполнено** | `protocol_matcher.annotate_applicability()` |
| Rules из summary | **Частично** | `summary_to_protocol_rules()` → `run_rule_checker(extra_rules=...)` |
| compliance_engine | **Выполнено** | `compliance_engine.build_compliance_report()` |
| Fallback legacy | **Частично** | Логика есть; на практике не срабатывает summary-ветка без match protocol_id |
| Hybrid сравнение | **Не выполнено** | `method_comparison` нигде не заполняется в `consult_analysis` |
| UI/API pipeline | **Частично** | `consult_review_pipeline.py:426` вызывает `analyze_consultation_text()` **без** `analysis_mode` → всегда env default (legacy) |

`merge_rules_for_plan()` из `method_selector.py` **не используется** в pipeline (dead code для hybrid merge).

---

## 11. CLI

| Команда | Статус | Пример вывода |
|---------|--------|---------------|
| `python -m scripts.build_protocol_summaries` | **Работает** | Полная сборка уже выполнена; `--help` OK |
| `python -m scripts.validate_protocol_summaries` | **Работает** | `akusherstvo_...: valid` / `needs_human_review` |
| `python -m scripts.export_protocol_summary_rules` | **Работает** | `Exported 12717 rules → .../exported_rules.json` |
| `python -m scripts.export_protocol_summary_rag` | **Работает** | `Wrote .../summary_chunks.jsonl` |
| `python -m scripts.compare_summary_vs_legacy` | **Работает** | `gastro_adult: legacy=85.6 summary=85.6 hybrid=85.6` |
| `python -m scripts.check_kz --mode legacy --file ...` | **Работает** | JSON compliance на stdout |
| `python -m scripts.check_kz --mode summary --file ...` | **Работает** | CLI принимает `--mode`, но результат **идентичен legacy** (summary не активируется) |
| `python -m scripts.check_kz --mode hybrid --file ...` | **Работает** | То же — score 85.6, `analysis_mode` в JSON отсутствует |

`check_kz` — обёртка над `analyze_consultation.main()` (`scripts/check_kz.py`).

---

## 12. Method comparison

| Поле ТЗ | Статус |
|---------|--------|
| `overall_score` / scores по режимам | **Частично** | `summary_compare.compare_modes_on_text()` — только legacy/summary/hybrid score |
| `confidence_score`, `matched_protocols`, issues, red_flags, explainability | **Не выполнено** |
| Отчёт comparison | **Частично** | `write_comparison_report()` — 5 строк (scores + delta); не `comparison_reports/` |

Пример (`data/reports/method_comparison/gastro_adult.md` после `--file tests/fixtures/consultations/gastro_adult.txt`):

```markdown
# Method comparison: gastro_adult

- legacy score: 85.6
- summary score: 85.6
- hybrid score: 85.6
- delta (summary-legacy): 0.0
- same decision: True
```

---

## 13. Method selector

| Сценарий | Статус | Проверка |
|----------|--------|----------|
| `mode=legacy` → legacy | **Выполнено** | `tests/test_method_selector.py::test_legacy_when_disabled` |
| `mode=summary` + valid summary | **Выполнено** | unit-тесты + ручная проверка с `matched_protocol_ids=['test_gastro_k30']` |
| summary invalid → fallback / insufficient | **Выполнено** | `resolve_analysis_plan` + тесты |
| hybrid summary primary + legacy | **Выполнено** | `test_hybrid_uses_summary_when_available` (fixtures) |

---

## 14. Compliance engine

| Пункт | Статус |
|-------|--------|
| Принимает legacy + summary rules | **Частично** | Через единый `rules_check` / `extra_rules` в `rule_checker`, не раздельные параметры |
| `rule_source` | **Выполнено** | `rule_model.ProtocolRule.rule_source`; summary rules → `"summary"` |
| Hybrid conflict priority | **Не выполнено** | `merge_rules_for_plan()` не вызывается; нет логики approved>table>legacy; нет auto `manual_review` при конфликте |

Поля summary в отчёте: `compliance_engine.build_compliance_report()` строки 523–535 — заполняет `analysis_mode`, `protocol_summary_used`, `fallback_to_legacy`.

---

## 15. Evidence map

Схема: `consult_schema.EvidenceMapItem`; построение: `evidence_map.build_evidence_map()`.

| Поле ТЗ | Статус |
|---------|--------|
| `rule_id`, `rule_source`, `required_item`, `found_*`, `consultation_evidence`, `protocol_evidence`, `decision`, `explanation` | **Выполнено** |
| `protocol_id`, `condition_id` | **Не выполнено** — полей нет в `EvidenceMapItem` |

### Пример (gastro_1, legacy)

```json
{
  "rule_id": "8e7327d9_auto_functional_dyspepsia_diagnosis_formula",
  "rule_source": "legacy",
  "rule_type": "diagnosis_structure_rule",
  "found_in_consultation": false,
  "decision": "missing",
  "explanation": "В формулировке диагноза не хватает компонентов...",
  "source_refs": [{ "local_path": "minzdrav_protocols/gastroenterologiya/КП_...185.pdf" }]
}
```

Summary-rules в evidence map на реальных КЗ **не появляются** (summary-ветка не активна).

---

## 16. Отчёты

### Markdown (`consult_report.report_to_markdown`)

| Требование | Статус |
|------------|--------|
| Режим оценки, Protocol Summary, статус карточки, fallback | **Частично** | Блок строк 607–614 — только если `analysis_mode != legacy` |
| Где summary vs legacy разошлись | **Не выполнено** |
| Evidence map | **Выполнено** |
| Source refs summary vs raw PDF | **Не выполнено** |
| Ограничения | **Выполнено** |

### JSON (`consult_report.report_to_json`)

| Поле | Статус |
|------|--------|
| `analysis_mode`, `protocol_summary_used`, `protocol_summary_status`, `fallback_to_legacy`, `legacy/summary_result_available`, `method_comparison`, `summary_source_refs`, `legacy_source_refs` | **Не выполнено** — поля есть в `ComplianceReport`, но **не сериализуются** в `report_to_json()` (строки 254–286) |

Проверка: `check_kz --mode hybrid` → `"analysis_mode": null` в stdout JSON.

---

## 17. Summary quality

Файл: `data/protocol_summaries/summary_quality_report.md` (генератор: `summary_quality.py`).

| Метрика ТЗ | Статус |
|------------|--------|
| protocols with summary | **Выполнено** — 475 |
| valid summaries | **Выполнено** — 240 |
| approved | **Выполнено** — 0 |
| conditions, ICD, exams, treatment, red_flags, rules, rules with quote | **Выполнено** |
| rules with source_ref | **Не выполнено** |
| доля table-derived rules | **Не выполнено** |

---

## 18. Unit-тесты

| Файл | Статус | Результат |
|------|--------|-----------|
| `tests/test_protocol_summary_schema.py` | **Выполнено** | 3 passed |
| `tests/test_protocol_summary_validator.py` | **Выполнено** | 3 passed |
| `tests/test_summary_to_rules.py` | **Выполнено** | 3 passed |
| `tests/test_summary_to_rag.py` | **Выполнено** | 1 passed |
| `tests/test_summary_loader.py` | **Выполнено** | 3 passed |
| `tests/test_method_selector.py` | **Выполнено** | 3 passed |
| `tests/test_summary_vs_legacy.py` | **Выполнено** | 3 passed |

**Запуск (2026-05-31):**
```text
25 passed in 0.54s   # protocol summary + regression KZ subset
~228 passed        # полный pytest -q (весь suite)
```

---

## 19. Regression-тесты на КЗ (3 режима)

Текстовые фикстуры: `tests/test_regression_kz_compliance.py` (gastro_1, mg_1, pl_1_f, pl_2_d_s); pl_1_d / pl_2_d_s_2 / pl_2_d_s_3 — только в ТЗ, не как отдельные файлы.

**Автопрогон legacy/summary/hybrid** (скрипт аудита, `PROTOCOL_SUMMARY_ENABLED=1` для summary/hybrid):

| file | mode | overall_score | confidence | status | matched_protocols | red_flags | fallback_used | critical_issues | summary_used |
|------|------|---------------|------------|--------|-------------------|-----------|---------------|-----------------|--------------|
| gastro_1 | legacy | 75.5 | 89.0 | mostly_compliant | 0 | 1 | — | 1 | — |
| gastro_1 | summary | 75.5 | 89.0 | mostly_compliant | 0 | 1 | — | 1 | — |
| gastro_1 | hybrid | 75.5 | 89.0 | mostly_compliant | 0 | 1 | — | 1 | — |
| mg_1 | legacy | 64.7 | 70.8 | partially_compliant | 0 | 0 | — | 0 | — |
| mg_1 | summary | 64.7 | 70.8 | partially_compliant | 0 | 0 | — | 0 | — |
| mg_1 | hybrid | 64.7 | 70.8 | partially_compliant | 0 | 0 | — | 0 | — |
| pl_1_f | legacy | 86.5 | 89.0 | mostly_compliant | 0 | 1 | — | 1 | — |
| pl_1_f | summary | 86.5 | 89.0 | mostly_compliant | 0 | 1 | — | 1 | — |
| pl_1_f | hybrid | 86.5 | 89.0 | mostly_compliant | 0 | 1 | — | 1 | — |
| pl_2_d_s | legacy | 71.7 | 89.0 | partially_compliant | 0 | 0 | — | 0 | — |
| pl_2_d_s | summary | 71.7 | 89.0 | partially_compliant | 0 | 0 | — | 0 | — |
| pl_2_d_s | hybrid | 71.7 | 89.0 | partially_compliant | 0 | 0 | — | 0 | — |
| pl_1_d | legacy | 54.5 | 79.0 | partially_compliant | 0 | 0 | — | 0 | — |
| pl_1_d | summary | 54.5 | 79.0 | partially_compliant | 0 | 0 | — | 0 | — |
| pl_1_d | hybrid | 54.5 | 79.0 | partially_compliant | 0 | 0 | — | 0 | — |
| pl_2_d_s_2 | legacy | 50.0 | 75.0 | partially_compliant | 0 | 0 | — | 0 | — |
| pl_2_d_s_2 | summary | 50.0 | 75.0 | partially_compliant | 0 | 0 | — | 0 | — |
| pl_2_d_s_2 | hybrid | 50.0 | 75.0 | partially_compliant | 0 | 0 | — | 0 | — |
| pl_2_d_s_3 | legacy | 71.7 | 89.0 | partially_compliant | 0 | 0 | — | 0 | — |
| pl_2_d_s_3 | summary | 71.7 | 89.0 | partially_compliant | 0 | 0 | — | 0 | — |
| pl_2_d_s_3 | hybrid | 71.7 | 89.0 | partially_compliant | 0 | 0 | — | 0 | — |

**Вывод:** все три режима дают **идентичные** scores — summary/hybrid фактически не меняют проверку. Причины: (1) `matched_protocols=0` в offline-тесте; (2) ICD-fallback не вызывается при `use_summary=False`; (3) `report_to_json` не экспортирует `protocol_summary_used`.

Отдельные regression-тесты парсера/safety (`test_regression_kz_compliance.py`) — **6 passed** (legacy-поведение парсера и safety_checker).

---

## 20. Медицинская логика (кейсы ТЗ)

| Кейс | Статус | Комментарий |
|------|--------|-------------|
| **gastro_1** — ICD, malignancy, suspected | **Частично** | Парсер + `safety_checker` (`possible_malignancy`) — **OK** (`test_gastro_1_icd_and_malignancy`). Summary red flag **не участвует** в online scoring. |
| **mg_1** — undefined | **Частично** | `has_undefined` в extraction_quality — **OK**; отдельный data_quality issue в compliance — слабо выражен. |
| **pl_1_d** — L30 и L93.0 отдельно, exams | **Не проверено** | Нет dedicated test; inline pl_1_d парсится, но required/conditional exams split не верифицирован. |
| **pl_1_f** — thrombosis, rivaroxaban, УЗИ 3 мес | **Выполнено** (legacy path) | `test_pl_1_f_thrombosis_and_medication` — thrombosis handled/partially_handled. |
| **pl_2_d_s** — suspected `?` | **Выполнено** | `test_pl_2_d_s_suspected_and_exams`. |
| **pl_2_d_s_2** — prednisolone schedule | **Не выполнено** | `parse_consultation` → `schedule_steps=0` для tapering prednisolone (MedicationScheduleStep не заполняется). |

---

## 21. Legacy не сломан

| Проверка | Статус |
|----------|--------|
| Полный pytest | **OK** — ~228 tests passed |
| `check_kz --mode legacy` | **OK** |
| `consult_review_pipeline` + RAG | **OK** — работает без `PROTOCOL_SUMMARY_ENABLED` |
| Default env | **OK** — summary disabled |

**Замечание:** structured analysis в UI (`CONSULT_STRUCTURED_ANALYSIS=True`) вызывает `analyze_consultation_text()` без summary mode — поведение совпадает с legacy baseline, регрессии нет.

---

## 22. Финальный вывод

### Полностью реализовано

- Pydantic schema и документация (`docs/protocol_summary_*.md`)
- Builder + batch corpus (478 PDF → drafts/yaml/json/chunks)
- Validator + validation reports
- Loader + summary→rules + summary→RAG export
- Method selector (unit-tested)
- Env-config с safe defaults (`ENABLED=0`, `MODE=legacy`)
- Unit-тесты модуля (25 tests)
- Legacy baseline сохранён

### Частично реализовано

- Режимы summary/hybrid (код есть, online не активирует summary на реальных КЗ)
- Online pipeline (`consult_analysis` — да; `consult_review_pipeline`/UI — без `analysis_mode`)
- Summary→rules (нет contraindication/applicability rule types)
- Отчёты (ComplianceReport model vs JSON export)
- Method comparison (только score delta)
- Summary quality report (не все метрики ТЗ)
- Хранение (`reviewed/`, `comparison_reports/` отсутствуют)
- Качество auto_extracted карточек (240/475 valid, 0 approved)
- Медицинские regression-кейсы через summary cards

### Не реализовано

- RAG retrieve по `summary_chunks` + hybrid boost
- `exporter.py` (в ТЗ; фактически scripts)
- `docs/protocol_summary_quality_criteria.md`
- Hybrid conflict resolution + `merge_rules_for_plan` в pipeline
- `method_comparison` / `summary_source_refs` / `legacy_source_refs` в JSON и MD
- Regression «3 режима дают различимый summary-эффект» на gastro_1…pl_2_d_s_3
- pl_2_d_s_2 MedicationScheduleStep
- EvidenceMapItem.`protocol_id` / `condition_id`

### Оставшиеся риски

1. **Иллюзия готовности:** CLI `--mode summary` принимается, но результат = legacy → ложное ощущение работы hybrid.
2. **protocol_id mismatch:** card matcher IDs ≠ summary `protocol_id` → summaries не подключаются даже при наличии ICD-match карточек.
3. **Draft quality:** heuristic builder, strict validation отсекает ~47% как invalid/needs_review.
4. **94 MB data в git** — operational, не functional risk.

### Следующие 5 задач (приоритет)

1. **Починить активацию summary:** ICD-based lookup до/вместо strict protocol_id match; вызывать ICD-fallback при `mode in (summary, hybrid)` даже если `use_summary=False`.
2. **Прокинуть `analysis_mode` в UI:** `consult_review_pipeline` + env/API toggle; заполнить `report_to_json` полями ТЗ §23.
3. **Подключить `summary_chunks` в `rag_server.retrieve()`** с boost в hybrid.
4. **Использовать `merge_rules_for_plan()`** + conflict → `manual_review` issue.
5. **Human review loop:** `reviewed/` folder, поднять `MIN_REVIEW_STATUS=reviewed` для prod pilot.

### Можно ли включать summary/hybrid в продакшн?

**Нет — пока оставить `legacy`.**

Обоснование:
- Default уже legacy — prod безопасен.
- Summary/hybrid **не меняют** scoring и evidence map на реальных КЗ.
- 0 approved cards; auto_extracted drafts не проходят медицинский review.
- RAG-слой summary не участвует в retrieve.

**Рекомендация:** pilot summary только после задач 1–2 + review ≥50 ключевых карточек (gastro, phleb, derma) с `review_status=reviewed`.

---

*Аудит выполнен без изменений кода. Для воспроизведения: `.venv/bin/pytest tests/test_protocol_summary*.py tests/test_summary*.py tests/test_method_selector.py` и `python -m scripts.compare_summary_vs_legacy --file tests/fixtures/consultations/gastro_adult.txt`.*
