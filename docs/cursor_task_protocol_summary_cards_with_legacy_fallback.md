# ТЗ для Cursor: Protocol Summary Cards для проверки КЗ по протоколам РБ с сохранением legacy-подхода

## 1. Контекст проекта

В проекте уже реализована рабочая архитектура проверки консультативных заключений:

- скачивание протоколов Минздрава РБ;
- парсинг PDF/DOC;
- построение корпуса чанков;
- карточки протоколов;
- каталог клинических правил;
- RAG-поиск;
- structured analysis;
- `compliance_engine`;
- `safety_checker`;
- гибридный итоговый процент;
- CLI;
- API/UI.

Эту архитектуру не удалять и не переписывать с нуля.

Нужно добавить новый слой: **Protocol Summary Cards** - подробные нормализованные выдержки из каждого протокола в едином формате. Новый слой должен стать основным источником правил для проверки КЗ, но старый подход должен остаться доступным как fallback, baseline и режим сравнения.

---

## 2. Главная идея новой методики

Сейчас приложение проверяет КЗ в основном через:

```text
PDF протокола
  -> текстовые чанки
  -> RAG
  -> правила, извлеченные из корпуса
  -> compliance engine
```

Нужно добавить более надежный слой:

```text
PDF протокола
  -> текст, секции, таблицы
  -> Protocol Summary YAML
  -> валидация
  -> Protocol Rules
  -> RAG chunks
  -> KZ compliance checker
  -> Evidence map
  -> Report
```

То есть каждый протокол должен быть преобразован в подробную нормализованную клиническую карточку.

Проверка КЗ должна идти не только по сырым чанкам PDF, а по структурированной карточке:

```text
КЗ
  -> диагнозы, МКБ, возраст, пол, жалобы, обследования, лечение
  -> подбор condition из Protocol Summary
  -> проверка applicability
  -> проверка diagnosis_structure
  -> проверка clinical_criteria
  -> проверка diagnostic_criteria
  -> проверка required_exams
  -> проверка treatment
  -> проверка follow_up
  -> проверка red_flags
  -> evidence map
  -> отчет
```

---

## 3. Важное требование: сохранить старую методику

Старый подход должен остаться доступным.

Нужно реализовать 3 режима работы:

```text
legacy
summary
hybrid
```

### 3.1 legacy

Использует текущий подход:

```text
protocol_cards + rules_from_corpus + RAG chunks + current compliance_engine
```

Этот режим нужен как baseline и fallback.

### 3.2 summary

Использует новый подход:

```text
Protocol Summary YAML/JSON -> generated rules -> compliance_engine
```

Этот режим должен стать основным после проверки качества.

### 3.3 hybrid

Комбинирует оба подхода:

```text
Protocol Summary rules - основной источник
Legacy RAG/rules - fallback и дополнительное evidence
LLM/RAG - только объяснение и цитаты, не повышение итогового score
```

Если summary-карточка отсутствует или не прошла валидацию, автоматически использовать legacy.

---

## 4. Новые env-переменные

Добавить env-переменные:

```text
PROTOCOL_SUMMARY_MODE=hybrid
PROTOCOL_SUMMARY_ENABLED=1
PROTOCOL_SUMMARY_STRICT_VALIDATION=1
PROTOCOL_SUMMARY_FALLBACK_TO_LEGACY=1
PROTOCOL_SUMMARY_COMPARE_WITH_LEGACY=1
PROTOCOL_SUMMARY_GENERATE_RULES=1
PROTOCOL_SUMMARY_GENERATE_RAG=1
PROTOCOL_SUMMARY_MIN_REVIEW_STATUS=draft
```

Допустимые значения:

```text
PROTOCOL_SUMMARY_MODE:
  legacy
  summary
  hybrid
```

Логика:

```text
legacy - использовать старый подход
summary - использовать только Protocol Summary, если карточка валидна
hybrid - использовать Protocol Summary как основной источник, legacy как fallback
```

---

## 5. Новая структура файлов

Добавить структуру:

```text
clinical_knowledge/
  protocol_summary/
    __init__.py
    schema.py
    builder.py
    validator.py
    loader.py
    exporter.py
    summary_to_rules.py
    summary_to_rag.py
    summary_compare.py
    summary_quality.py
    method_selector.py
    prompts/
      protocol_summary_extraction_prompt.md
      protocol_summary_review_prompt.md

data/
  protocol_summaries/
    yaml/
    json/
    drafts/
    reviewed/
    validation_reports/
    comparison_reports/

docs/
  protocol_summary_schema.md
  protocol_summary_workflow.md
  protocol_summary_quality_criteria.md
```

Если в проекте уже есть аналогичные папки, использовать существующую структуру и не дублировать модули.

---

## 6. Форматы хранения

Источник истины:

```text
data/protocol_summaries/yaml/{protocol_id}.yaml
```

Машиночитаемый экспорт:

```text
data/protocol_summaries/json/{protocol_id}.json
```

Черновики:

```text
data/protocol_summaries/drafts/{protocol_id}.yaml
```

Проверенные карточки:

```text
data/protocol_summaries/reviewed/{protocol_id}.yaml
```

Отчеты валидации:

```text
data/protocol_summaries/validation_reports/{protocol_id}.md
```

Сравнение со старым подходом:

```text
data/protocol_summaries/comparison_reports/{protocol_id}.md
```

---

## 7. Protocol Summary Schema

Создать pydantic-схему в:

```text
clinical_knowledge/protocol_summary/schema.py
```

### 7.1 ProtocolSummary

```python
class ProtocolSummary(BaseModel):
    protocol_id: str
    summary_version: str = "1.0"

    extraction_status: Literal[
        "draft",
        "auto_extracted",
        "needs_human_review",
        "reviewed",
        "deprecated"
    ] = "draft"

    review_status: Literal[
        "not_reviewed",
        "needs_review",
        "reviewed",
        "approved",
        "rejected"
    ] = "not_reviewed"

    source: ProtocolSource
    rubric: ProtocolRubric
    applicability: ProtocolApplicability

    conditions: list[ConditionSummary]

    global_red_flags: list[RedFlagItem] = []
    global_contraindications: list[ContraindicationItem] = []
    global_notes: list[SummaryNote] = []

    extraction_metadata: ExtractionMetadata
    validation: SummaryValidationResult | None = None
```

### 7.2 ProtocolSource

```python
class ProtocolSource(BaseModel):
    title: str
    url: str | None = None
    local_path: str | None = None
    document_sha256: str | None = None

    approval_date: str | None = None
    approval_number: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    document_year: int | None = None

    pages_total: int | None = None
```

### 7.3 ProtocolRubric

```python
class ProtocolRubric(BaseModel):
    name: str
    slug: str | None = None
    specialty_slugs: list[str] = []
```

### 7.4 ProtocolApplicability

```python
class ProtocolApplicability(BaseModel):
    population: list[Literal[
        "adult",
        "child",
        "newborn",
        "pregnant",
        "adult_and_child",
        "unknown"
    ]] = []

    age_min_years: int | None = None
    age_max_years: int | None = None

    sex: Literal["male", "female", "any", "unknown"] = "unknown"
    pregnancy: Literal["required", "excluded", "any", "unknown"] = "unknown"

    care_setting: list[Literal[
        "outpatient",
        "inpatient",
        "emergency",
        "intensive_care",
        "rehabilitation",
        "palliative",
        "unknown"
    ]] = []
```

### 7.5 ConditionSummary

```python
class ConditionSummary(BaseModel):
    condition_id: str
    name: str
    synonyms: list[str] = []
    abbreviations: list[str] = []

    icd10_codes: list[str] = []

    condition_applicability: ProtocolApplicability | None = None

    diagnosis_structure: DiagnosisStructure | None = None
    clinical_criteria: CriteriaBlock | None = None
    diagnostic_criteria: CriteriaBlock | None = None

    required_exams: list[ExamRequirement] = []
    conditional_exams: list[ExamRequirement] = []

    treatment: TreatmentBlock | None = None
    follow_up: list[FollowUpRequirement] = []

    hospitalization: list[RoutingRequirement] = []
    routing: list[RoutingRequirement] = []

    red_flags: list[RedFlagItem] = []
    contraindications: list[ContraindicationItem] = []
    complications: list[ComplicationItem] = []

    kz_checklist: KzChecklist | None = None

    source_refs: list[SourceRef] = []
```

### 7.6 SourceRef

```python
class SourceRef(BaseModel):
    protocol_id: str
    document_url: str | None = None
    local_path: str | None = None

    page_start: int | None = None
    page_end: int | None = None

    section_title: str | None = None
    section_type: str | None = None

    table_index: int | None = None
    row_index: int | None = None

    quote: str | None = None
```

Правило: любой clinical item должен иметь `source_ref`.

---

## 8. Блоки карточки

### 8.1 DiagnosisStructure

```python
class DiagnosisStructure(BaseModel):
    required_components: list[DiagnosisComponent] = []
    optional_components: list[DiagnosisComponent] = []
    examples: list[DiagnosisExample] = []
    source_refs: list[SourceRef] = []
```

```python
class DiagnosisComponent(BaseModel):
    name: str
    required: bool = True
    description: str | None = None
    source_ref: SourceRef
```

Примеры компонентов:

```text
нозология
клиническая форма
стадия
степень тяжести
фаза
активность
локализация
осложнения
сопутствующие состояния
```

### 8.2 CriteriaBlock

```python
class CriteriaBlock(BaseModel):
    required: list[CriterionItem] = []
    optional: list[CriterionItem] = []
    exclusion: list[CriterionItem] = []
```

```python
class CriterionItem(BaseModel):
    text: str
    logic_group: str | None = None
    operator: Literal[
        "present",
        "absent",
        "contains",
        "any_of",
        "all_of",
        "numeric_gte",
        "numeric_lte",
        "duration_gte",
        "frequency_gte",
        "unknown"
    ] = "unknown"

    values: list[str] = []
    numeric_value: float | None = None
    unit: str | None = None

    evidence_targets: list[Literal[
        "complaints",
        "anamnesis",
        "objective_status",
        "local_status",
        "performed_exams",
        "recommended_exams",
        "diagnosis",
        "treatment",
        "follow_up"
    ]] = []

    source_ref: SourceRef
```

### 8.3 ExamRequirement

```python
class ExamRequirement(BaseModel):
    name: str
    aliases: list[str] = []

    exam_type: Literal[
        "laboratory",
        "instrumental",
        "imaging",
        "functional",
        "consultation",
        "pathology",
        "unknown"
    ] = "unknown"

    requirement_level: Literal[
        "required",
        "conditional",
        "recommended",
        "optional"
    ]

    accepted_statuses: list[Literal[
        "performed",
        "recommended",
        "planned",
        "control"
    ]] = ["performed", "recommended"]

    required_if: list[str] = []
    timing: str | None = None
    comment: str | None = None

    source_ref: SourceRef
```

### 8.4 TreatmentBlock

```python
class TreatmentBlock(BaseModel):
    non_drug: list[NonDrugTreatmentItem] = []
    drug_groups: list[DrugGroupItem] = []
    drugs: list[DrugTreatmentItem] = []
    procedures: list[ProcedureTreatmentItem] = []
    surgery: list[SurgeryTreatmentItem] = []
    source_refs: list[SourceRef] = []
```

```python
class DrugTreatmentItem(BaseModel):
    drug_name: str | None = None
    active_substance: str | None = None
    drug_group: str | None = None

    dose_text: str | None = None
    frequency_text: str | None = None
    duration_text: str | None = None
    route: str | None = None

    indication: str | None = None
    contraindications: list[str] = []
    monitoring: list[str] = []

    applicability: ProtocolApplicability | None = None
    source_ref: SourceRef
```

### 8.5 FollowUpRequirement

```python
class FollowUpRequirement(BaseModel):
    text: str
    timing: str | None = None
    required_if: list[str] = []
    expected_actions: list[str] = []
    source_ref: SourceRef
```

### 8.6 RedFlagItem

```python
class RedFlagItem(BaseModel):
    text: str
    aliases: list[str] = []

    red_flag_type: Literal[
        "possible_malignancy",
        "thrombosis",
        "severe_infection",
        "systemic_autoimmune",
        "drug_safety",
        "urgent_referral",
        "other"
    ]

    severity: Literal["low", "medium", "high", "critical"]

    expected_actions: list[str] = []
    cap_if_unhandled: int | None = None

    source_ref: SourceRef
```

### 8.7 KzChecklist

```python
class KzChecklist(BaseModel):
    must_have: list[str] = []
    should_have: list[str] = []
    conditional: list[str] = []
    warnings: list[str] = []
```

---

## 9. YAML-формат карточки

Пример целевого YAML:

```yaml
protocol_id: gastro_k30_functional_dyspepsia_2025
summary_version: "1.0"
extraction_status: draft
review_status: not_reviewed

source:
  title: "Клинический протокол ..."
  url: "https://..."
  local_path: "minzdrav_protocols/gastro/..."
  approval_date: "2025-11-11"
  approval_number: "185"
  valid_from: "2026-02-04"
  document_year: 2025
  pages_total: 120

rubric:
  name: "Гастроэнтерология"
  slug: "gastroenterologiya"
  specialty_slugs:
    - gastroenterology

applicability:
  population:
    - adult
  age_min_years: 18
  age_max_years: null
  sex: any
  pregnancy: any
  care_setting:
    - outpatient
    - inpatient

conditions:
  - condition_id: k30_functional_dyspepsia
    name: "Функциональная диспепсия"
    synonyms:
      - "диспепсия"
      - "функциональная диспепсия"
    abbreviations: []
    icd10_codes:
      - K30

    diagnosis_structure:
      required_components:
        - name: "нозология"
          required: true
          description: "Указать нозологию"
          source_ref:
            protocol_id: gastro_k30_functional_dyspepsia_2025
            page_start: 10
            section_title: "Формулировка диагноза"
            quote: "..."
      optional_components: []
      examples:
        - text: "Функциональная диспепсия, смешанный вариант"
          source_ref:
            protocol_id: gastro_k30_functional_dyspepsia_2025
            page_start: 10
            quote: "..."

    clinical_criteria:
      required:
        - text: "наличие диспепсических жалоб"
          operator: present
          evidence_targets:
            - complaints
            - anamnesis
          source_ref:
            protocol_id: gastro_k30_functional_dyspepsia_2025
            page_start: 12
            section_title: "Клинические критерии"
            quote: "..."
      optional: []
      exclusion: []

    diagnostic_criteria:
      required:
        - text: "исключение органической патологии"
          operator: present
          evidence_targets:
            - performed_exams
            - recommended_exams
          source_ref:
            protocol_id: gastro_k30_functional_dyspepsia_2025
            page_start: 13
            section_title: "Диагностические критерии"
            quote: "..."
      optional: []
      exclusion: []

    required_exams:
      - name: "ЭГДС"
        aliases:
          - "эзофагогастродуоденоскопия"
        exam_type: instrumental
        requirement_level: required
        accepted_statuses:
          - performed
          - recommended
        required_if:
          - "наличие диспепсических жалоб"
        timing: null
        source_ref:
          protocol_id: gastro_k30_functional_dyspepsia_2025
          page_start: 14
          section_title: "Диагностика"
          quote: "..."

    conditional_exams:
      - name: "УЗИ органов брюшной полости"
        aliases:
          - "УЗИ ОБП"
        exam_type: imaging
        requirement_level: conditional
        accepted_statuses:
          - performed
          - recommended
        required_if:
          - "подозрение на билиарную патологию"
        source_ref:
          protocol_id: gastro_k30_functional_dyspepsia_2025
          page_start: 15
          quote: "..."

    treatment:
      non_drug:
        - text: "диетические рекомендации"
          source_ref:
            protocol_id: gastro_k30_functional_dyspepsia_2025
            page_start: 20
            quote: "..."
      drug_groups:
        - drug_group: "прокинетики"
          indication: "при симптомах диспепсии"
          source_ref:
            protocol_id: gastro_k30_functional_dyspepsia_2025
            page_start: 21
            quote: "..."
      drugs: []

    follow_up:
      - text: "повторная консультация после дообследования"
        timing: null
        required_if:
          - "назначено дообследование"
        expected_actions:
          - "оценить результаты обследований"
        source_ref:
          protocol_id: gastro_k30_functional_dyspepsia_2025
          page_start: 24
          quote: "..."

    red_flags:
      - text: "подозрение на опухолевое образование"
        aliases:
          - "опухолевое образование"
          - "нельзя исключить инвазию"
        red_flag_type: possible_malignancy
        severity: critical
        expected_actions:
          - "дообследование"
          - "маршрутизация"
          - "профильная консультация"
        cap_if_unhandled: 45
        source_ref:
          protocol_id: gastro_k30_functional_dyspepsia_2025
          page_start: 30
          quote: "..."

    kz_checklist:
      must_have:
        - diagnosis
        - complaints
        - anamnesis
        - objective_status
        - performed_or_recommended_exams
        - recommendations
      should_have:
        - follow_up
      conditional:
        - routing_if_red_flag
      warnings:
        - "подозрительный диагноз без дообследования"
```

---

## 10. Protocol Summary Builder

Создать:

```text
clinical_knowledge/protocol_summary/builder.py
```

Задачи:

```text
1. Читать output/documents/*.json, output/chunks/chunks.jsonl и таблицы.
2. Для каждого protocol_id создавать draft YAML.
3. Если один PDF содержит несколько заболеваний - создавать несколько conditions внутри одного ProtocolSummary.
4. Извлекать:
   - МКБ;
   - нозологии;
   - синонимы;
   - возрастную применимость;
   - пол;
   - беременность;
   - условия помощи;
   - формулу диагноза;
   - клинические критерии;
   - диагностические критерии;
   - обязательные обследования;
   - дополнительные обследования;
   - лечение;
   - дозы;
   - длительность;
   - немедикаментозные рекомендации;
   - follow-up;
   - госпитализацию;
   - маршрутизацию;
   - противопоказания;
   - осложнения;
   - красные флаги.
5. Сохранять source_ref для каждого элемента.
```

Первый этап может быть эвристическим. Если в проекте доступен LLM, разрешается использовать его для draft-экстракции, но результат обязательно валидировать.

---

## 11. Prompt для автоматической выдержки

Создать:

```text
clinical_knowledge/protocol_summary/prompts/protocol_summary_extraction_prompt.md
```

Промпт должен требовать:

```text
- не выдумывать данные;
- сохранять только то, что есть в протоколе;
- указывать source_ref;
- если данных нет - оставлять пустое поле;
- не давать медицинских рекомендаций от себя;
- разделять нозологии;
- разделять required и conditional;
- отделять performed/recommended будет уже на уровне КЗ, не протокола;
- коротко цитировать источник.
```

---

## 12. Protocol Summary Validator

Создать:

```text
clinical_knowledge/protocol_summary/validator.py
```

Проверки:

```text
1. protocol_id заполнен.
2. source.title заполнен.
3. source.url или local_path заполнен.
4. rubric.name заполнен.
5. applicability.population заполнен или unknown.
6. conditions не пустой.
7. У каждой condition есть condition_id.
8. У каждой condition есть name.
9. У каждой condition есть icd10_codes или reason почему нет.
10. У каждого clinical item есть source_ref.
11. У каждого source_ref есть page_start или section_title.
12. Нет clinical item без цитаты, если STRICT_VALIDATION=1.
13. Нет дублирующихся conditions.
14. Нет дублирующихся exam requirements.
15. Нет дублирующихся drug items.
16. Required и conditional не смешаны.
17. Age applicability не противоречит population.
18. Не слишком длинные quote.
19. Нет пустых expected_actions у critical red flag.
20. Все YAML валидны по pydantic-схеме.
```

Выход:

```text
data/protocol_summaries/validation_reports/{protocol_id}.md
```

Валидационный статус:

```text
valid
valid_with_warnings
invalid
needs_human_review
```

---

## 13. Human review workflow

Добавить workflow для ручной проверки карточек.

Статусы:

```text
draft
auto_extracted
needs_human_review
reviewed
approved
rejected
deprecated
```

Минимальная логика:

```text
- Автоматически созданные карточки получают review_status=not_reviewed.
- После валидации с предупреждениями - needs_review.
- Если эксперт проверил - reviewed или approved.
- Для production по умолчанию использовать карточки со статусом draft и выше, но в строгом режиме только reviewed/approved.
```

Настройка:

```text
PROTOCOL_SUMMARY_MIN_REVIEW_STATUS=draft
```

Допустимые значения:

```text
draft
reviewed
approved
```

---

## 14. Summary to Rules

Создать:

```text
clinical_knowledge/protocol_summary/summary_to_rules.py
```

Задача: генерировать `ProtocolRule` из `ProtocolSummary`.

Маппинг:

```text
diagnosis_structure.required_components -> diagnosis_structure_rule
clinical_criteria.required -> clinical_criterion_rule
diagnostic_criteria.required -> diagnostic_criterion_rule
required_exams -> required_exam_rule
conditional_exams -> conditional_exam_rule
treatment.drug_groups -> treatment_group_rule
treatment.drugs -> drug_rule / drug_dose_rule / drug_duration_rule
treatment.non_drug -> non_drug_rule
follow_up -> follow_up_rule
routing -> routing_rule
red_flags -> red_flag_rule
contraindications -> contraindication_rule
applicability -> age/sex/pregnancy applicability rules
```

Каждое правило должно получить:

```text
rule_id
protocol_id
condition_id
condition_name
icd10_codes
rule_type
severity
applicability
evidence_targets
expected_items
criteria
source_ref
generated_from_summary=true
summary_id
summary_version
```

---

## 15. Summary to RAG

Создать:

```text
clinical_knowledge/protocol_summary/summary_to_rag.py
```

Задача: генерировать RAG-чанки не только из сырого PDF, но и из нормализованной карточки.

Типы чанков:

```text
summary_overview
summary_diagnosis_structure
summary_clinical_criteria
summary_diagnostic_criteria
summary_required_exams
summary_treatment
summary_follow_up
summary_red_flags
summary_contraindications
```

Каждый summary chunk должен иметь:

```text
chunk_id
protocol_id
condition_id
condition_name
icd10_codes
rubric_name
section_type
text
source_refs
generated_from_summary=true
```

RAG должен уметь искать по двум корпусам:

```text
raw_pdf_chunks
summary_chunks
```

В режиме hybrid summary_chunks должны иметь boost.

---

## 16. Summary Loader

Создать:

```text
clinical_knowledge/protocol_summary/loader.py
```

Функции:

```python
load_protocol_summaries() -> list[ProtocolSummary]
load_summary_by_protocol_id(protocol_id: str) -> ProtocolSummary | None
find_conditions_by_icd(icd10_code: str) -> list[ConditionSummary]
find_conditions_by_text(query: str) -> list[ConditionSummary]
load_summary_rules() -> list[ProtocolRule]
```

---

## 17. Интеграция в online pipeline

В `consult_review_pipeline.py` добавить новую ветку.

Логика:

```text
1. Распарсить КЗ.
2. Извлечь диагнозы, МКБ, возраст, пол, беременность.
3. Найти Protocol Summary conditions.
4. Проверить applicability.
5. Сгенерировать или загрузить rules из summary.
6. Запустить compliance_engine по summary rules.
7. Если summary отсутствует или invalid:
   - использовать legacy pipeline.
8. Если режим hybrid:
   - использовать summary result как основной;
   - legacy result как fallback evidence;
   - сравнить различия.
```

---

## 18. Интеграция в CLI

Добавить команды:

```bash
python -m scripts.build_protocol_summaries
python -m scripts.validate_protocol_summaries
python -m scripts.export_protocol_summary_rules
python -m scripts.export_protocol_summary_rag
python -m scripts.compare_summary_vs_legacy
python -m scripts.check_kz --mode summary --file kz.pdf
python -m scripts.check_kz --mode legacy --file kz.pdf
python -m scripts.check_kz --mode hybrid --file kz.pdf
```

Параметры:

```text
--protocol-id
--rubric
--limit
--strict
--review-status
--output
```

---

## 19. Сравнение новой и старой методики

Создать:

```text
clinical_knowledge/protocol_summary/summary_compare.py
```

Задача: сравнить результаты `legacy`, `summary`, `hybrid` на одном КЗ и на наборе КЗ.

Сравнивать:

```text
overall_score
confidence_score
matched_protocols
matched_conditions
critical_issues
major_issues
missing_required_exams
treatment_issues
red_flags
manual_review_required
source_refs_count
explainability_score
```

Выход:

```text
data/reports/method_comparison/{consultation_id}.md
data/reports/method_comparison/batch_comparison.csv
```

Добавить метрики:

```text
summary_better
legacy_better
same_decision
different_decision
needs_manual_review
```

Важно: если новый подход хуже на части кейсов, проект должен позволять откатиться на legacy или hybrid без изменения кода.

---

## 20. Method selector

Создать модуль:

```text
clinical_knowledge/protocol_summary/method_selector.py
```

Логика выбора:

```text
if mode == legacy:
    use legacy
elif mode == summary:
    use summary if summary valid else fail or fallback depending on env
elif mode == hybrid:
    use summary if valid, legacy as fallback/evidence
```

Если summary invalid:

```text
- если PROTOCOL_SUMMARY_FALLBACK_TO_LEGACY=1 -> legacy;
- иначе -> insufficient_protocol_data.
```

---

## 21. Обновить compliance_engine

`compliance_engine` должен принимать rules из двух источников:

```text
legacy_rules
summary_rules
```

Добавить поле:

```python
rule_source: Literal["legacy", "summary", "manual", "table", "llm_draft"]
```

В отчете показывать:

```text
- правило из summary;
- правило из legacy;
- правило из table;
- правило из manual review.
```

В режиме hybrid при конфликте:

```text
1. reviewed/approved summary rule имеет приоритет.
2. table-derived summary rule имеет приоритет над keyword legacy rule.
3. legacy используется как fallback, если summary не покрывает раздел.
4. Если конфликт существенный - issue manual_review.
```

---

## 22. Обновить Evidence map

Evidence map должен показывать источник правила.

```python
class EvidenceMapItem(BaseModel):
    rule_id: str
    rule_source: Literal["summary", "legacy", "table", "manual", "llm_draft"]
    protocol_id: str
    condition_id: str | None = None

    required_item: str | None = None
    found_in_consultation: bool

    found_status: Literal[
        "performed",
        "recommended",
        "mentioned",
        "not_found",
        "not_applicable",
        "unknown"
    ]

    consultation_evidence: list[str] = []
    protocol_evidence: list[str] = []

    decision: Literal[
        "satisfied",
        "satisfied_by_recommendation",
        "missing",
        "not_applicable",
        "manual_review",
        "unknown"
    ]

    explanation: str
```

---

## 23. Обновить отчеты

Markdown-отчет должен показывать:

```text
1. Режим оценки: legacy / summary / hybrid.
2. Использованы ли Protocol Summary Cards.
3. Статус карточек: draft / reviewed / approved.
4. Какие протоколы покрыты summary.
5. Где был fallback на legacy.
6. Где summary и legacy дали разные выводы.
7. Evidence map.
8. Source refs из summary.
9. Source refs из raw PDF.
10. Ограничения оценки.
```

JSON-отчет должен содержать:

```json
{
  "analysis_mode": "hybrid",
  "protocol_summary_used": true,
  "protocol_summary_status": "draft",
  "fallback_to_legacy": false,
  "legacy_result_available": true,
  "summary_result_available": true,
  "method_comparison": {
    "same_decision": true,
    "score_delta": 4.2,
    "critical_issue_delta": 0
  },
  "evidence_map": [],
  "summary_source_refs": [],
  "legacy_source_refs": []
}
```

---

## 24. Валидация качества новой методики

Добавить `summary_quality.py`.

Проверять на уровне всего корпуса:

```text
- сколько протоколов имеют summary;
- сколько summary валидны;
- сколько approved;
- сколько conditions извлечено;
- сколько МКБ покрыто;
- сколько required_exams извлечено;
- сколько treatment items извлечено;
- сколько red_flags извлечено;
- сколько rules сгенерировано;
- сколько rules имеют source_ref;
- сколько rules имеют quote;
- доля table-derived rules.
```

Отчет:

```text
data/protocol_summaries/summary_quality_report.md
```

---

## 25. Regression-тесты

Добавить тесты:

```text
tests/test_protocol_summary_schema.py
tests/test_protocol_summary_validator.py
tests/test_summary_to_rules.py
tests/test_summary_to_rag.py
tests/test_summary_loader.py
tests/test_method_selector.py
tests/test_summary_vs_legacy.py
```

Проверки:

```text
1. YAML проходит pydantic validation.
2. Clinical item без source_ref дает ошибку.
3. Condition без name дает ошибку.
4. Required exam превращается в required_exam_rule.
5. Red flag превращается в red_flag_rule.
6. Drug item превращается в drug_rule.
7. Follow-up item превращается в follow_up_rule.
8. Summary chunks создаются.
9. Loader находит condition по МКБ.
10. Method selector выбирает legacy, если summary invalid.
11. Hybrid использует summary как основной источник.
12. Summary и legacy comparison сохраняет отчет.
```

---

## 26. Regression-тесты на реальных КЗ

Использовать обезличенные КЗ в:

```text
tests/fixtures/consultations/
```

Проверить на каждом КЗ три режима:

```text
legacy
summary
hybrid
```

Минимальные ожидания:

### gastro_1

```text
- извлекаются K30, Q43.8, E61.1, E80.4, R14, K82.8;
- опухолевое образование -> possible_malignancy;
- "нельзя исключить инвазию" -> suspected + critical red flag;
- колоноскопия под седацией -> recommended_exam;
- summary или hybrid не должен терять critical red flag;
- если summary отсутствует по онкологической маршрутизации, hybrid должен использовать legacy/safety fallback.
```

### mg_1

```text
- undefined -> data_quality issue;
- J06/J06.8 извлекаются;
- если summary по ОРИ отсутствует, режим hybrid должен fallback на legacy;
- low confidence допустим, но не silent success.
```

### pl_1_d

```text
- L30 и L93.0 извлекаются отдельно;
- шаблонные блоки >>> распознаются;
- обязательные обследования распознаются как recommended required exams;
- summary rules должны проверить required_exams.
```

### pl_1_f

```text
- I80.1 извлекается;
- флеботромбоз -> thrombosis red flag;
- ривароксабан 20 мг 1 раз в день -> medication;
- контроль УЗИ через 3 месяца -> handled safety action.
```

### pl_2_d_s и pl_2_d_s_2

```text
- дискоидная красная волчанка ? -> suspected;
- ANA/anti-DNA -> recommended_exams;
- гидроксихлорохин -> medication;
- преднизолон schedule -> MedicationScheduleStep[];
- drug_safety flag обработан через follow-up/план снижения дозы.
```

---

## 27. Критерии приемки

Задача считается выполненной, если:

```text
1. Старый режим legacy продолжает работать.
2. Добавлен режим summary.
3. Добавлен режим hybrid.
4. Добавлена pydantic-схема ProtocolSummary.
5. YAML summary валидируется.
6. JSON export работает.
7. Summary builder создает draft YAML хотя бы для выбранных протоколов.
8. Summary validator создает validation report.
9. Summary to rules генерирует ProtocolRule.
10. Summary to RAG генерирует summary chunks.
11. Loader находит condition по МКБ.
12. Online pipeline может использовать summary rules.
13. CLI поддерживает --mode legacy/summary/hybrid.
14. Если summary invalid, работает fallback на legacy.
15. Evidence map показывает rule_source.
16. Markdown report показывает режим анализа.
17. JSON report показывает protocol_summary_used и fallback_to_legacy.
18. Method comparison сохраняет отчет.
19. Summary quality report формируется.
20. Regression-тесты проходят.
21. Неприменимые по возрасту/полу/беременности правила не снижают оценку.
22. Red flags не теряются при переходе на summary.
23. При отсутствии summary система не падает.
24. При конфликте summary и legacy выводится manual_review issue.
25. LLM не повышает deterministic score.
```

---

## 28. Порядок реализации

Реализовать строго по этапам.

### Этап 1 - план

```text
Создать docs/protocol_summary_implementation_plan.md
Описать, какие текущие модули будут расширены
Описать риски
Описать fallback strategy
```

### Этап 2 - schema

```text
Создать clinical_knowledge/protocol_summary/schema.py
Создать docs/protocol_summary_schema.md
Добавить тесты схемы
```

### Этап 3 - validator

```text
Создать validator.py
Добавить validation reports
Добавить тесты
```

### Этап 4 - loader

```text
Создать loader.py
Реализовать поиск по protocol_id, condition_id, МКБ и тексту
```

### Этап 5 - builder

```text
Создать builder.py
Сделать draft generation из существующих documents/chunks/tables
Не требовать идеального качества на первом этапе
```

### Этап 6 - summary_to_rules

```text
Создать summary_to_rules.py
Интегрировать rule_source
```

### Этап 7 - summary_to_rag

```text
Создать summary_to_rag.py
Добавить summary chunks
```

### Этап 8 - method_selector

```text
Создать method_selector.py
Интегрировать legacy/summary/hybrid
```

### Этап 9 - compliance integration

```text
Обновить compliance_engine
Поддержать legacy_rules и summary_rules
Добавить conflict handling
```

### Этап 10 - reports

```text
Обновить JSON/Markdown reports
Добавить method comparison
Добавить summary quality report
```

### Этап 11 - tests

```text
Добавить unit tests
Добавить regression tests на реальные КЗ
Проверить legacy не сломан
```

---

## 29. Первая команда для Cursor

После добавления этого файла в проект выполнить:

```text
Изучи docs/cursor_task_protocol_summary_cards_with_legacy_fallback.md.
Не переписывай проект с нуля.
Сохрани текущий legacy-пайплайн как baseline и fallback.
Сначала создай docs/protocol_summary_implementation_plan.md с планом безопасной интеграции.
Затем реализуй по этапам:
1. schema;
2. validator;
3. loader;
4. builder;
5. summary_to_rules;
6. summary_to_rag;
7. method_selector;
8. compliance integration;
9. reports;
10. tests.
После каждого этапа проверь, что legacy-режим продолжает работать.
```
