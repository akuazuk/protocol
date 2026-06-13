# Cursor Task: Clinical Protocols and Consultation Compliance Analyzer

## 0. Назначение документа

Этот файл является основным техническим заданием для Cursor.

Нужно усовершенствовать существующий проект или создать недостающие модули так, чтобы приложение могло:

1. Собирать и структурировать клинические протоколы Минздрава Республики Беларусь.
2. Парсить консультативные заключения врачей.
3. Извлекать из КЗ возраст, пол, диагнозы, МКБ, жалобы, анамнез, обследования, назначения, дозировки, длительность лечения, повторные явки и другие параметры.
4. Подбирать релевантные протоколы с учетом возраста, пола, беременности, специальности, диагноза, МКБ и даты консультации.
5. Давать оценку соответствия КЗ протоколам.
6. Формировать проверяемый отчет с баллами, замечаниями, предупреждениями и ссылками на источники.

Приложение не должно ставить диагноз и не должно заменять врача. Оно должно быть инструментом экспертной проверки документа на соответствие клиническим протоколам.

---

## 1. Источники клинических протоколов

Базовая страница:

```text
https://minzdrav.gov.by/ru/dlya-spetsialistov/standarty-obsledovaniya-i-lecheniya/
```

Нужно обработать все рубрики протоколов:

1. Акушерство, гинекология
2. Аллергология, иммунология
3. Анестезиология, реаниматология
4. Болезни системы кровообращения
5. Гастроэнтерология
6. Гематология
7. Дерматовенерология
8. Заболевания перинатального периода
9. Инфекционные заболевания
10. Неврология, нейрохирургия
11. Нефрология
12. Новообразования
13. Оториноларингология
14. Офтальмология
15. Паллиативная помощь
16. Психиатрия, наркология
17. Пульмонология, фтизиатрия
18. Ревматология
19. Стоматология
20. Травматология, ортопедия
21. Трансплантация органов и тканей
22. Урология
23. Хирургия
24. Эндокринология, нарушения обмена веществ

Если на сайте появятся дополнительные рубрики, проект должен уметь их обнаружить автоматически и добавить в отчет.

---

## 2. Важный порядок работы Cursor

Если проект уже существует, сначала выполнить аудит.

### Шаг 1. Аудит текущего проекта

Перед написанием нового кода Cursor должен:

1. Изучить структуру проекта.
2. Изучить презентацию проекта, если она есть в `docs/`.
3. Найти текущие модели данных.
4. Найти текущий пайплайн работы с протоколами.
5. Найти текущий пайплайн работы с консультативными заключениями.
6. Найти текущий UI/API, если есть.
7. Найти текущие тесты.
8. Найти текущие ограничения.
9. Составить файл:

```text
docs/current_project_audit.md
```

В `docs/current_project_audit.md` указать:

```text
Что уже реализовано
Что работает
Что не работает
Какие модули нужно сохранить
Какие модули нужно переписать
Какие модули нужно добавить
Какие изменения могут сломать текущую функциональность
План безопасного рефакторинга
```

Важно: не удалять рабочую функциональность без необходимости. Если уже есть API, UI или база данных, новые модули интегрировать в текущую архитектуру, а не создавать параллельное несовместимое приложение.

### Шаг 2. План внедрения

После аудита создать:

```text
docs/implementation_plan.md
```

План должен быть поэтапным:

1. Модели данных.
2. Парсер КЗ.
3. Парсер протоколов.
4. Подбор протоколов.
5. Compliance engine.
6. Scoring.
7. Отчеты.
8. Тесты.
9. Интеграция с существующим UI/API.

---

## 3. Рекомендуемая структура проекта

Если текущая структура проекта позволяет, добавить следующие файлы и папки:

```text
clinical_protocols_project/
  README.md
  requirements.txt
  pyproject.toml
  .env.example
  .gitignore

  docs/
    cursor_task_protocols_and_consultations.md
    current_project_audit.md
    implementation_plan.md
    project_presentation.pdf

  config/
    settings.yaml
    rubrics.yaml
    section_types.yaml
    rule_types.yaml
    extraction_schema.yaml
    consultation_schema.yaml
    compliance_weights.yaml
    consultation_section_patterns.yaml
    medication_patterns.yaml
    red_flags.yaml

  data/
    raw/
      html/
      documents/
    examples/
      consultations/
    interim/
      crawled_links.jsonl
      downloaded_documents.jsonl
      parsed_pages.jsonl
      parsed_tables.jsonl
    processed/
      protocols_registry.jsonl
      protocol_sections.jsonl
      protocol_entities.jsonl
      protocol_rules.jsonl
      chunks_for_rag.jsonl
      consultation_examples_schema.json
    reports/
      crawl_report.md
      download_report.md
      parsing_report.md
      quality_report.md
      rubric_summary.md

  logs/
    pipeline.log
    errors.log

  src/
    __init__.py
    main.py
    settings.py
    models.py
    crawler.py
    downloader.py
    html_parser.py
    pdf_parser.py
    docx_parser.py
    text_cleaner.py
    section_splitter.py
    section_classifier.py
    table_extractor.py
    metadata_extractor.py
    entity_extractor.py
    rule_extractor.py
    rag_chunker.py
    validators.py
    reporting.py
    utils.py

    consultation/
      __init__.py
      consultation_models.py
      consultation_parser.py
      consultation_normalizer.py
      consultation_fact_extractor.py
      consultation_diagnosis_matcher.py
      protocol_matcher.py
      compliance_engine.py
      scoring.py
      evidence_builder.py
      report_generator.py
      longitudinal_analyzer.py
      safety_checker.py
      medication_parser.py
      date_parser.py
      age_sex_resolver.py

  tests/
    test_crawler.py
    test_downloader.py
    test_pdf_parser.py
    test_text_cleaner.py
    test_section_classifier.py
    test_models.py
    test_validators.py
    test_consultation_parser.py
    test_diagnosis_parser.py
    test_medication_parser.py
    test_protocol_matcher.py
    test_compliance_engine.py
    fixtures/
      consultations/
```

Если текущий проект уже имеет другую структуру, не ломать ее. Интегрировать модули аккуратно.

---

## 4. Технологические требования

Использовать Python 3.11 или выше.

Рекомендуемые библиотеки:

```text
requests
beautifulsoup4
lxml
pydantic
pydantic-settings
pyyaml
pandas
tqdm
pymupdf
pdfplumber
python-docx
python-dateutil
regex
tenacity
rich
pytest
```

Общие требования:

1. Использовать `pathlib.Path`.
2. Использовать type hints.
3. Использовать pydantic-модели для основных сущностей.
4. Все JSONL-файлы писать в UTF-8.
5. Все ошибки логировать.
6. Ошибка одного документа или одного КЗ не должна останавливать весь batch-процесс.
7. Все этапы должны сохранять промежуточный результат.
8. Должна быть защита от повторного скачивания.
9. Должна быть проверка дублей по URL и SHA256.
10. Все выводы приложения должны иметь source refs.

---

## 5. CLI-команды

Добавить или расширить CLI:

```bash
python -m src.main crawl
python -m src.main download
python -m src.main parse
python -m src.main split-sections
python -m src.main classify-sections
python -m src.main extract-metadata
python -m src.main extract-entities
python -m src.main extract-rules
python -m src.main build-rag
python -m src.main validate
python -m src.main report
python -m src.main all

python -m src.main parse-consultation --file path/to/consultation.pdf
python -m src.main analyze-consultation --file path/to/consultation.pdf
python -m src.main analyze-consultation-folder --folder data/examples/consultations
python -m src.main evaluate-consultations --folder data/examples/consultations
```

Поддержать параметры:

```bash
--rubric "Гастроэнтерология"
--limit 10
--force
--skip-existing
--verbose
--output data/reports
```

---

## 6. Конфигурация

Создать или обновить `config/settings.yaml`.

```yaml
base_url: "https://minzdrav.gov.by"
start_url: "https://minzdrav.gov.by/ru/dlya-spetsialistov/standarty-obsledovaniya-i-lecheniya/"

paths:
  raw_html: "data/raw/html"
  raw_documents: "data/raw/documents"
  example_consultations: "data/examples/consultations"
  interim: "data/interim"
  processed: "data/processed"
  reports: "data/reports"
  logs: "logs"

crawler:
  timeout_seconds: 30
  retries: 3
  user_agent: "ClinicalProtocolsResearchBot/1.0"
  allowed_file_extensions:
    - ".pdf"
    - ".doc"
    - ".docx"
    - ".rtf"

parser:
  min_text_chars_per_page: 50
  detect_ocr_required: true
  extract_tables: true

chunking:
  min_chars: 800
  max_chars: 5000
  target_chars: 2500
  overlap_chars: 300

validation:
  fail_on_critical: false
```

Создать `config/compliance_weights.yaml`.

```yaml
weights:
  protocol_match_score: 0.15
  diagnosis_score: 0.20
  required_exams_score: 0.20
  treatment_score: 0.20
  safety_score: 0.15
  documentation_quality_score: 0.10

status_thresholds:
  compliant: 90
  mostly_compliant: 75
  partially_compliant: 50
  non_compliant: 1
```

Создать `config/red_flags.yaml`.

```yaml
red_flags:
  possible_malignancy:
    keywords:
      - опухолевое образование
      - нельзя исключить инвазию
      - подозрение на злокачественное
      - образование кишки
    severity: critical
    expected_actions:
      - дообследование
      - консультация профильного специалиста
      - маршрутизация
      - повторная консультация

  thrombosis:
    keywords:
      - флеботромбоз
      - тромбоз глубоких вен
      - ТГВ
    severity: high
    expected_actions:
      - антикоагулянтная терапия
      - контроль УЗИ
      - повторная консультация

  systemic_autoimmune:
    keywords:
      - дискоидная красная волчанка
      - системная красная волчанка
      - ANA
      - anti-DNA
    severity: medium
    expected_actions:
      - лабораторное обследование
      - консультация ревматолога по показаниям
      - фотозащита
```

---

## 7. Модели данных протоколов

Создать или обновить `src/models.py`.

### SourceRef

```python
class SourceRef(BaseModel):
    document_url: str
    normalized_document_url: str | None = None
    local_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    section_type: str | None = None
    quote: str | None = None
```

### ProtocolRegistryItem

```python
class ProtocolRegistryItem(BaseModel):
    protocol_id: str
    rubric_id: str
    rubric_name: str
    rubric_url: str
    document_title: str
    document_url: str
    normalized_document_url: str
    file_type: str | None = None
    source_page_url: str
    local_path: str | None = None

    crawl_datetime: datetime
    download_datetime: datetime | None = None
    download_status: str | None = None
    download_error: str | None = None

    file_size_bytes: int | None = None
    mime_type: str | None = None
    sha256: str | None = None

    approval_date: str | None = None
    approval_number: str | None = None
    valid_from: str | None = None
    document_year: int | None = None
    status: str | None = None

    population: str | None = None
    care_setting: list[str] = []
    icd10_codes: list[str] = []
```

### ProtocolRule

```python
class ProtocolRule(BaseModel):
    rule_id: str
    protocol_id: str
    rubric_name: str
    document_title: str

    condition_name: str | None = None
    icd10_codes: list[str] = []

    applicable_age_groups: list[str] = []
    applicable_sex: Literal["male", "female", "any", "unknown"] = "unknown"
    pregnancy_applicability: Literal[
        "pregnant_only",
        "not_pregnant",
        "any",
        "unknown"
    ] = "unknown"

    rule_type: str
    severity: str
    logic: str

    trigger_conditions: list[dict] = []
    criteria: list[dict] = []
    expected_items: list[str] = []
    forbidden_items: list[str] = []

    evidence_field_targets: list[str] = []

    source: SourceRef
```

### ApplicabilityFilter

```python
class ApplicabilityFilter(BaseModel):
    age_min_years: int | None = None
    age_max_years: int | None = None
    age_groups: list[str] = []

    sex: Literal["male", "female", "any", "unknown"] = "unknown"
    pregnancy: Literal["required", "excluded", "any", "unknown"] = "unknown"

    care_setting: list[str] = []
    specialty: list[str] = []
    condition_status: list[str] = []
```

---

## 8. Сбор и структурирование протоколов

### 8.1 Crawler

`src/crawler.py` должен:

1. Открыть стартовую страницу.
2. Найти ссылки на рубрики.
3. Для каждой рубрики открыть страницу.
4. Сохранить HTML.
5. Найти ссылки на документы.
6. Определить тип файла.
7. Нормализовать ссылки.
8. Исключить дубли.
9. Создать `ProtocolRegistryItem`.

Выходные файлы:

```text
data/interim/crawled_links.jsonl
data/processed/protocols_registry.jsonl
```

### 8.2 Downloader

`src/downloader.py` должен:

1. Скачать документы.
2. Сохранить в `data/raw/documents`.
3. Вычислить SHA256.
4. Определить MIME type.
5. Записать статус скачивания.

Статусы:

```text
success
failed
skipped_existing
unsupported_type
empty_file
```

### 8.3 PDF/DOCX parser

Извлекать:

- текст по страницам;
- таблицы;
- количество страниц;
- метаданные;
- признак `ocr_required`.

Если на странице меньше 50 символов текста, но страница не пустая, ставить:

```json
"ocr_required": true
```

OCR на первом этапе не делать.

### 8.4 Section splitter

Преобразовать страницы в логические разделы.

Определять заголовки:

```text
ГЛАВА
Общие положения
Термины и определения
Классификация
Формулировка диагноза
Клинические критерии
Диагностические критерии
Лечение
Диагностика
Показания
Противопоказания
Госпитализация
Реабилитация
Диспансерное наблюдение
Профилактика
Приложение
```

### 8.5 Section classifier

Допустимые `section_type`:

```text
legal_metadata
approval_metadata
general_provisions
scope
terms_and_definitions
abbreviations
icd10
condition_definition
epidemiology
etiology
pathogenesis
classification
diagnosis_formula
clinical_criteria
diagnostic_criteria
differential_diagnosis
required_exams
conditional_exams
laboratory_tests
instrumental_tests
functional_tests
imaging
consultations
treatment_general
drug_treatment
non_drug_treatment
surgery
emergency_care
hospitalization
rehabilitation
follow_up
prevention
contraindications
complications
red_flags
quality_indicators
appendix
table
references
unknown
```

### 8.6 Entity extractor

Извлекать:

```text
condition_name
condition_abbrev
icd10_codes
diagnosis_components
classification_items
clinical_criteria
diagnostic_criteria
required_exams
conditional_exams
laboratory_tests
instrumental_tests
consultations
treatment_items
drug_treatment_items
non_drug_treatment_items
hospitalization_criteria
follow_up_items
contraindications
complications
red_flags
```

Важно: один PDF может содержать несколько нозологий. Не считать, что один PDF = одно заболевание.

### 8.7 Rule extractor

Типы правил:

```text
diagnosis_structure_rule
clinical_criterion_rule
diagnostic_criterion_rule
required_exam_rule
conditional_exam_rule
laboratory_test_rule
instrumental_test_rule
consultation_rule
treatment_rule
drug_treatment_rule
drug_dose_rule
non_drug_treatment_rule
hospitalization_rule
follow_up_rule
contraindication_rule
red_flag_rule
quality_rule
```

`severity`:

```text
required
recommended
conditional
warning
forbidden
informational
```

Определение `severity`:

- `required`: должен, необходимо, обязательно, проводится, назначается, выполняется, критериями являются, диагноз устанавливается, показано.
- `recommended`: рекомендуется, целесообразно, следует рассмотреть.
- `conditional`: при наличии, при подозрении, по показаниям, в случае, при необходимости, может быть.
- `warning`: противопоказано, с осторожностью, не рекомендуется, риск, осложнение.
- `forbidden`: запрещено, не допускается, не назначать, исключить применение.

Каждое правило должно иметь source ref.

### 8.8 RAG chunks

Создать:

```text
data/processed/chunks_for_rag.jsonl
```

Правила чанкинга:

1. Не резать внутри клинического правила.
2. Не резать внутри таблицы.
3. Не отделять заголовок от текста.
4. Не смешивать разные нозологии, если их можно разделить.
5. Не смешивать разные типы разделов.
6. Каждый чанк должен иметь source ref.

---

## 9. Модели данных консультативного заключения

Создать `src/consultation/consultation_models.py`.

### ConsultationDocument

```python
class ConsultationDocument(BaseModel):
    consultation_id: str
    source_file: str
    source_file_type: str
    raw_text: str
    pages: list[dict] = []

    clinic_name: str | None = None
    doctor_specialty: str | None = None
    doctor_name: str | None = None
    doctor_category: str | None = None

    consultation_date: date | None = None
    consultation_datetime: datetime | None = None

    patient: PatientContext
    sections: ConsultationSections
    diagnoses: list[ConsultationDiagnosis]
    medications: list[MedicationItem]
    planned_exams: list[ExamItem]
    performed_exams: list[ExamItem]
    follow_up: list[FollowUpItem]

    extraction_quality: ExtractionQuality
```

### PatientContext

```python
class PatientContext(BaseModel):
    full_name: str | None = None
    birth_date: date | None = None
    age_years: int | None = None
    age_months: int | None = None
    sex: Literal["male", "female", "unknown"] = "unknown"

    adult_or_child: Literal["adult", "child", "newborn", "unknown"] = "unknown"
    pregnancy: bool | None = None

    height_cm: float | None = None
    weight_kg: float | None = None
    bmi: float | None = None

    allergies: list[str] = []
    current_medications: list[str] = []
    comorbidities: list[str] = []
    surgeries: list[str] = []
    family_history: list[str] = []
    social_history: list[str] = []

    vitals: dict = {}
```

Возраст рассчитывать на дату консультации. Если дата рождения есть, но дата консультации не распознана, использовать текущую дату только как fallback и ставить warning.

Возрастные группы:

```text
newborn: 0-28 дней
infant: 29 дней - 1 год
child: 1-17 лет
adult: 18 лет и старше
elderly: 65 лет и старше
unknown
```

### ConsultationSections

```python
class ConsultationSections(BaseModel):
    complaints: str | None = None
    anamnesis: str | None = None
    allergy_history: str | None = None
    medication_history: str | None = None
    surgical_history: str | None = None
    objective_status: str | None = None
    local_status: str | None = None
    exam_results: str | None = None
    diagnosis_text: str | None = None
    recommendations_exams: str | None = None
    recommendations_treatment: str | None = None
    general_recommendations: str | None = None
    follow_up_text: str | None = None
```

Распознавать заголовки:

```text
Жалобы:
Анамнез:
Aнамнез:
Аллергия на лс:
Аллергоанамнез:
Объективный статус:
Локальный статус:
Данные обследований:
Диагноз:
Рекомендации:
Рекомендации по обследованию:
Рекомендации по лечению:
Дата повторной явки:
```

### ConsultationDiagnosis

```python
class ConsultationDiagnosis(BaseModel):
    diagnosis_id: str
    raw_text: str
    icd10_code: str | None = None
    diagnosis_name: str | None = None

    diagnosis_role: Literal[
        "primary",
        "secondary",
        "comorbidity",
        "symptom",
        "suspected",
        "finding",
        "unknown"
    ] = "unknown"

    certainty: Literal[
        "confirmed",
        "suspected",
        "excluded",
        "unclear"
    ] = "unclear"

    is_protocol_relevant: bool = True
    source_section: str | None = None
```

Правила:

- Если есть знак вопроса, `подозрение`, `нельзя исключить`, `вероятно`, `под вопросом`, то `certainty = suspected`.
- Если есть `основной`, то `diagnosis_role = primary`.
- Если есть `undefined`, добавить предупреждение качества.
- Анализировать несколько диагнозов в одном КЗ отдельно.

### ExamItem

```python
class ExamItem(BaseModel):
    exam_id: str
    exam_name: str
    exam_type: Literal[
        "laboratory",
        "instrumental",
        "imaging",
        "functional",
        "consultation",
        "pathology",
        "unknown"
    ] = "unknown"

    status: Literal[
        "performed",
        "recommended",
        "planned",
        "control",
        "unknown"
    ] = "unknown"

    date: date | None = None
    result_text: str | None = None
    result_value: str | None = None
    abnormal_flag: bool | None = None
    source_section: str | None = None
```

Разделять:

- выполненные обследования;
- рекомендованные обследования;
- контрольные обследования;
- обязательные обследования по протоколу;
- дополнительные обследования по протоколу;
- обследования по показаниям.

### MedicationItem

```python
class MedicationItem(BaseModel):
    medication_id: str
    raw_text: str

    drug_name: str | None = None
    active_substance: str | None = None
    dose_value: float | None = None
    dose_unit: str | None = None

    route: str | None = None
    frequency: str | None = None
    duration: str | None = None

    start_date: date | None = None
    end_date: date | None = None

    schedule: list[MedicationScheduleStep] = []
    indication_text: str | None = None
    source_section: str | None = None
```

```python
class MedicationScheduleStep(BaseModel):
    start_date: date | None = None
    end_date: date | None = None
    dose_text: str
    frequency_text: str | None = None
    daily_dose_text: str | None = None
```

Распознавать примеры:

```text
Тримедат форте 1 т 2 раза в день 28 дней
Эспумизан 2 к 3 раза в день по требованию
УДХК 2 капс вечером перед сном 3 месяца
Ривороксабан 20 мг раз в день постоянно
Гидроксихлорохин 200 мг по 1 таблетке 2 раза/сутки 2 недели
Преднизолон 5 мг по 12 таб в сутки
С 12.08.24 - Преднизолон 5 мг по 11 таб в сутки
С 19.08.24 - Преднизолон 5 мг по 10 таб в сутки
```

Длинные схемы снижения дозы сохранять как `schedule`.

### ExtractionQuality

```python
class ExtractionQuality(BaseModel):
    raw_text_length: int
    parsed_sections_count: int
    confidence: float

    warnings: list[str] = []
    errors: list[str] = []

    has_undefined: bool = False
    has_question_mark_diagnosis: bool = False
    has_unparsed_medication_schedule: bool = False
    has_missing_birth_date: bool = False
    has_missing_consultation_date: bool = False
    has_missing_doctor_specialty: bool = False
```

---

## 10. Парсер консультативных заключений

Создать `src/consultation/consultation_parser.py`.

Он должен принимать PDF/DOCX/TXT/JSON и возвращать `ConsultationDocument`.

Нужно извлекать:

1. Название клиники.
2. Специальность врача.
3. ФИО врача.
4. Категорию врача.
5. Дату и время консультации.
6. ФИО пациента.
7. Дату рождения.
8. Возраст на дату консультации.
9. Пол пациента.
10. Жалобы.
11. Анамнез.
12. Аллергоанамнез.
13. Лекарственный анамнез.
14. Хирургический анамнез.
15. Объективный статус.
16. Локальный статус.
17. Данные обследований.
18. Диагнозы.
19. МКБ-коды.
20. Рекомендации по обследованию.
21. Рекомендации по лечению.
22. Лекарственные назначения.
23. Дозировки.
24. Длительность.
25. Схемы снижения дозы.
26. Дату повторной явки.
27. Красные флаги.
28. Проблемы качества документа.

---

## 11. Обработка шаблонных КЗ

Некоторые КЗ содержат автошаблон:

```text
>>> L30 Экзема кожи ?:
* ОБСЛЕДОВАНИЯ ОБЯЗАТЕЛЬНЫЕ:
...
>>> L93.0 Дискоидная красная волчанка:
* ОБСЛЕДОВАНИЯ ДОПОЛНИТЕЛЬНЫЕ:
...
```

Добавить модель:

```python
class TemplateBlock(BaseModel):
    block_diagnosis_text: str
    icd10_code: str | None
    block_type: Literal[
        "required_exams",
        "additional_exams",
        "follow_up",
        "treatment",
        "care",
        "unknown"
    ]
    items: list[str]
```

Шаблонные блоки должны попадать в:

- diagnoses;
- planned_exams;
- treatment;
- follow_up;
- recommendations.

---

## 12. ProtocolMatcher

Создать `src/consultation/protocol_matcher.py`.

### Вход

```python
class ProtocolMatchInput(BaseModel):
    patient: PatientContext
    doctor_specialty: str | None
    diagnoses: list[ConsultationDiagnosis]
    complaints: list[str]
    exams: list[ExamItem]
    consultation_date: date | None
```

### Логика подбора

Использовать сигналы:

1. МКБ-код.
2. Текст диагноза.
3. Специальность врача.
4. Жалобы.
5. Обследования.
6. Возрастная группа.
7. Пол.
8. Беременность.
9. Дата консультации и дата действия протокола.
10. Рубрика протокола.

### Выход

```python
class ProtocolMatchResult(BaseModel):
    protocol_id: str
    rubric_name: str
    document_title: str
    matched_condition: str | None
    match_score: float
    match_reasons: list[str]
    mismatch_reasons: list[str]
    applicability: Literal[
        "applicable",
        "possibly_applicable",
        "not_applicable",
        "unknown"
    ]
    source_refs: list[SourceRef]
```

Важные правила:

1. Если МКБ совпал, но протокол детский, а пациент взрослый - не применять автоматически.
2. Если протокол только для беременных, а беременность не указана - не применять автоматически.
3. Если диагноз без МКБ, подбирать по тексту.
4. Если диагноз со знаком вопроса, применять правила как для `suspected`, а не как для подтвержденного заболевания.
5. Если есть несколько диагнозов, анализировать каждый отдельно.
6. Если есть красный флаг, он должен повысить приоритет соответствующего протокола или маршрутизации.

---

## 13. ComplianceEngine

Создать `src/consultation/compliance_engine.py`.

### Вход

```python
class ComplianceInput(BaseModel):
    consultation: ConsultationDocument
    matched_protocols: list[ProtocolMatchResult]
    protocol_rules: list[ProtocolRule]
    protocol_entities: list[ProtocolEntity]
```

### Выход

```python
class ComplianceReport(BaseModel):
    consultation_id: str
    overall_score: float | None
    overall_status: Literal[
        "compliant",
        "mostly_compliant",
        "partially_compliant",
        "non_compliant",
        "insufficient_data",
        "manual_review_required"
    ]

    protocol_matches: list[ProtocolMatchResult]
    diagnosis_assessments: list[DiagnosisAssessment]
    section_quality: SectionQualityAssessment
    exam_assessments: list[ExamAssessment]
    treatment_assessments: list[TreatmentAssessment]
    safety_assessments: list[SafetyAssessment]
    routing_assessments: list[RoutingAssessment]

    missing_required_items: list[ComplianceIssue]
    warnings: list[ComplianceIssue]
    critical_issues: list[ComplianceIssue]

    explanation: str
    source_refs: list[SourceRef]
```

---

## 14. Оценка диагноза

```python
class DiagnosisAssessment(BaseModel):
    diagnosis_id: str
    diagnosis_text: str
    icd10_code: str | None
    matched_condition: str | None
    matched_protocol_id: str | None

    status: Literal[
        "supported",
        "partially_supported",
        "not_supported",
        "suspected_needs_confirmation",
        "insufficient_data",
        "not_assessed"
    ]

    issues: list[ComplianceIssue]
    evidence_found: list[str]
    evidence_missing: list[str]
    source_refs: list[SourceRef]
```

Проверять:

1. Есть ли МКБ-код.
2. Соответствует ли МКБ тексту диагноза.
3. Есть ли обязательные компоненты диагноза: нозология, форма, степень, фаза, осложнения, локализация, стадия, активность, сопутствующие состояния.
4. Подтвержден ли диагноз жалобами, анамнезом, объективными данными и обследованиями.
5. Если диагноз подозрительный - есть ли назначено дообследование.
6. Если есть красный флаг - есть ли маршрутизация.

---

## 15. Оценка обследований

```python
class ExamAssessment(BaseModel):
    protocol_rule_id: str | None
    exam_name: str
    exam_type: str
    status: Literal[
        "present_performed",
        "present_recommended",
        "missing_required",
        "missing_conditional",
        "not_applicable",
        "extra_not_assessed"
    ]

    reason: str
    consultation_evidence: list[str]
    protocol_evidence: list[str]
    source_refs: list[SourceRef]
```

Проверять:

1. Обязательные обследования по протоколу.
2. Дополнительные обследования по показаниям.
3. Уже выполненные обследования.
4. Рекомендованные обследования.
5. Контрольные обследования.
6. Сроки обследований, если они указаны.
7. Наличие обследований перед процедурой или седацией.
8. Наличие лабораторного контроля при лекарственной терапии, если требуется.

---

## 16. Оценка лечения

```python
class TreatmentAssessment(BaseModel):
    medication_id: str | None
    treatment_text: str
    matched_protocol_rule_id: str | None

    status: Literal[
        "matches_protocol",
        "partially_matches_protocol",
        "not_in_protocol",
        "dose_mismatch",
        "duration_mismatch",
        "frequency_mismatch",
        "age_contraindication",
        "contraindication_warning",
        "insufficient_data",
        "not_assessed"
    ]

    issues: list[ComplianceIssue]
    protocol_evidence: list[str]
    consultation_evidence: list[str]
    source_refs: list[SourceRef]
```

Проверять:

1. Назначен ли препарат или группа препаратов, предусмотренные протоколом.
2. Соответствует ли доза.
3. Соответствует ли кратность.
4. Соответствует ли длительность.
5. Есть ли возрастные ограничения.
6. Есть ли противопоказания.
7. Есть ли необходимость контроля анализов.
8. Есть ли необходимость повторной консультации.
9. Есть ли немедикаментозные рекомендации, если они обязательны.
10. Есть ли назначения, не относящиеся к диагнозу.

---

## 17. SafetyChecker

Создать `src/consultation/safety_checker.py`.

```python
class SafetyAssessment(BaseModel):
    issue_type: Literal[
        "red_flag",
        "urgent_referral",
        "possible_malignancy",
        "thrombosis",
        "severe_infection",
        "drug_safety",
        "missing_control",
        "manual_review_required"
    ]

    severity: Literal["low", "medium", "high", "critical"]
    finding_text: str
    expected_action: str | None
    actual_action: str | None
    status: Literal[
        "handled",
        "partially_handled",
        "not_handled",
        "not_assessed"
    ]
    source_refs: list[SourceRef]
```

Примеры красных флагов:

- опухолевое образование;
- нельзя исключить инвазию;
- флеботромбоз;
- системная красная волчанка или подозрение на системное заболевание;
- высокая температура;
- выраженная боль;
- анемия/железодефицит с возможным источником кровопотери;
- подозрение на злокачественное новообразование;
- подозрение на осложнение лекарственной терапии.

Если есть красный флаг, проверить:

- дообследование;
- маршрутизацию;
- повторную консультацию;
- срочность;
- контроль;
- консультацию профильного специалиста.

Если есть критический красный флаг и нет маршрутизации, итоговый статус не должен быть выше:

```text
manual_review_required
```

---

## 18. Оценка качества КЗ

```python
class SectionQualityAssessment(BaseModel):
    has_complaints: bool
    has_anamnesis: bool
    has_objective_status: bool
    has_exam_results: bool
    has_diagnosis: bool
    has_recommendations: bool
    has_treatment: bool
    has_follow_up: bool

    missing_sections: list[str]
    duplicate_sections: list[str]
    suspicious_placeholders: list[str]
    extraction_warnings: list[str]
```

Проверять:

1. Есть ли обязательные разделы.
2. Нет ли пустых разделов.
3. Нет ли `undefined`.
4. Нет ли повторяющихся диагнозов без объяснения.
5. Нет ли назначений без дозировки.
6. Нет ли назначений без длительности, если длительность критична.
7. Нет ли неструктурированных фрагментов, которые не удалось разобрать.
8. Есть ли дата повторной явки, если она нужна.
9. Есть ли врач и специальность.
10. Есть ли дата консультации.

---

## 19. Scoring

Создать `src/consultation/scoring.py`.

```python
class ScoreBreakdown(BaseModel):
    protocol_match_score: float | None
    diagnosis_score: float | None
    required_exams_score: float | None
    treatment_score: float | None
    safety_score: float | None
    documentation_quality_score: float | None
    overall_score: float | None
```

Рекомендуемые веса:

```text
protocol_match_score: 15%
diagnosis_score: 20%
required_exams_score: 20%
treatment_score: 20%
safety_score: 15%
documentation_quality_score: 10%
```

Статусы:

```text
90-100: compliant
75-89: mostly_compliant
50-74: partially_compliant
1-49: non_compliant
0 или невозможно оценить: insufficient_data
есть критический красный флаг: manual_review_required или critical_issue
```

Если данные недостаточны, не ставить искусственно низкий балл. Возвращать:

```json
"overall_status": "insufficient_data"
```

---

## 20. Итоговые отчеты

Создать `src/consultation/report_generator.py`.

Формировать два формата:

```text
.json
.md
```

### JSON-отчет

```json
{
  "consultation_id": "example_consultation",
  "patient_summary": {
    "age_years": 48,
    "sex": "male",
    "adult_or_child": "adult"
  },
  "doctor_specialty": "Дерматолог",
  "consultation_date": "2024-07-14",
  "matched_protocols": [],
  "score_breakdown": {
    "protocol_match_score": null,
    "diagnosis_score": null,
    "required_exams_score": null,
    "treatment_score": null,
    "safety_score": null,
    "documentation_quality_score": null,
    "overall_score": null
  },
  "overall_status": "manual_review_required",
  "critical_issues": [],
  "warnings": [],
  "missing_required_items": [],
  "diagnosis_assessments": [],
  "exam_assessments": [],
  "treatment_assessments": [],
  "safety_assessments": [],
  "source_refs": []
}
```

### Markdown-отчет

Структура:

```text
# Оценка консультативного заключения

## 1. Краткое резюме
- Пациент
- Возраст
- Пол
- Дата консультации
- Специальность врача
- Основные диагнозы
- Подобранные протоколы
- Общая оценка

## 2. Применимость протокола
- Почему выбран протокол
- Почему протокол применим или неприменим
- Возрастные и половые ограничения

## 3. Оценка диагноза
- Что подтверждено
- Что не подтверждено
- Какие компоненты диагноза отсутствуют

## 4. Оценка обследований
- Выполненные обследования
- Рекомендованные обследования
- Отсутствующие обязательные обследования

## 5. Оценка лечения
- Назначения, соответствующие протоколу
- Назначения с неполной информацией
- Несоответствия дозировки, кратности или длительности
- Возрастные и другие ограничения

## 6. Красные флаги и безопасность
- Найденные риски
- Проверка маршрутизации
- Нужно ли ручное рассмотрение

## 7. Качество оформления КЗ
- Пропущенные разделы
- Дубли
- Placeholder-значения
- Неструктурированные данные

## 8. Ссылки на источники
- Протокол
- Страница
- Раздел
- Цитата
```

---

## 21. Работа с повторными консультациями

Создать `src/consultation/longitudinal_analyzer.py`.

Если загружено несколько КЗ одного пациента, приложение должно уметь объединить их в эпизод лечения.

```python
class EpisodeOfCare(BaseModel):
    episode_id: str
    patient_key: str
    specialty: str | None = None
    condition_candidates: list[str] = []
    consultations: list[ConsultationDocument] = []
    timeline: list[dict] = []
```

Анализировать:

1. Динамику жалоб.
2. Динамику диагноза.
3. Динамику лечения.
4. Выполнены ли назначенные ранее обследования.
5. Явился ли пациент на повторный прием.
6. Изменялась ли терапия обоснованно.
7. Есть ли снижение дозы по плану.
8. Есть ли контроль безопасности терапии.

---

## 22. Специальные требования по всем рубрикам

### 22.1 Акушерство, гинекология

Извлекать: беременность, срок беременности, роды, послеродовый период, риски для матери и плода, показания к госпитализации, показания к оперативному вмешательству, маршрутизацию пациентки.

### 22.2 Аллергология, иммунология

Извлекать: аллерген, тип реакции, анафилаксию, иммунодефицит, кожные пробы, IgE, противопоказания, неотложную помощь.

### 22.3 Анестезиология, реаниматология

Извлекать: интенсивную терапию, критические состояния, шкалы тяжести, мониторинг, вентиляцию, инфузионную терапию, неотложные действия.

### 22.4 Болезни системы кровообращения

Извлекать: артериальное давление, сердечную недостаточность, ИБС, инфаркт, ЭКГ, ЭхоКГ, липидный профиль, антикоагулянты, антиагреганты, показания к госпитализации.

### 22.5 Гастроэнтерология

Извлекать: жалобы ЖКТ, эндоскопические критерии, H. pylori, заболевания пищевода, желудка, кишечника, печени, желчного пузыря, поджелудочной железы, диетические рекомендации, длительность терапии.

### 22.6 Гематология

Извлекать: анемии, гемобластозы, коагулопатии, общий анализ крови, ферритин, коагулограмму, трансфузии, показания к госпитализации.

### 22.7 Дерматовенерология

Извлекать: кожные проявления, локализацию высыпаний, ИППП, лабораторное подтверждение, наружное лечение, системное лечение, фотозащиту, противопоказания.

### 22.8 Заболевания перинатального периода

Извлекать: новорожденного, срок гестации, массу тела, врожденные состояния, перинатальные инфекции, интенсивную терапию новорожденных, критерии риска.

### 22.9 Инфекционные заболевания

Извлекать: возбудителя, путь передачи, эпидемиологический анамнез, ПЦР, серологию, антибактериальную терапию, противовирусную терапию, изоляцию, профилактику.

### 22.10 Неврология, нейрохирургия

Извлекать: неврологический дефицит, шкалы, нейровизуализацию, ЭЭГ, инсульт, эпилепсию, показания к нейрохирургии, реабилитацию.

### 22.11 Нефрология

Извлекать: СКФ, креатинин, протеинурию, альбуминурию, мочевой синдром, ХБП, диализ, нефропротективную терапию.

### 22.12 Новообразования

Извлекать: локализацию опухоли, стадию, TNM, морфологию, гистологию, маркеры, химиотерапию, лучевую терапию, хирургическое лечение, диспансерное наблюдение.

### 22.13 Оториноларингология

Извлекать: ЛОР-симптомы, слух, носовое дыхание, эндоскопию, аудиометрию, показания к операции.

### 22.14 Офтальмология

Извлекать: остроту зрения, внутриглазное давление, глазное дно, поля зрения, офтальмоскопию, OCT, глаукому, катаракту, неотложные состояния.

### 22.15 Паллиативная помощь

Извлекать: боль, шкалы боли, функциональный статус, уход, маршрутизацию, обезболивание, поддерживающую терапию.

### 22.16 Психиатрия, наркология

Извлекать: психический статус, риск суицида, зависимости, интоксикации, абстиненцию, госпитализацию, психофармакотерапию, противопоказания.

### 22.17 Пульмонология, фтизиатрия

Извлекать: кашель, одышку, спирометрию, сатурацию, рентген, КТ, туберкулез, мокроту, антибактериальную терапию, противотуберкулезную терапию.

### 22.18 Ревматология

Извлекать: суставной синдром, воспалительную активность, аутоантитела, СРБ, СОЭ, DAS28 и другие шкалы, базисную терапию, генно-инженерные препараты.

### 22.19 Стоматология

Извлекать: зубную формулу, воспалительные заболевания полости рта, кариес, пародонтит, хирургическую стоматологию, обезболивание, профилактику.

### 22.20 Травматология, ортопедия

Извлекать: травму, переломы, вывихи, локализацию повреждения, рентген, КТ, иммобилизацию, показания к операции, реабилитацию.

### 22.21 Трансплантация органов и тканей

Извлекать: орган, донорство, реципиента, иммунологическую совместимость, иммуносупрессию, противопоказания, мониторинг после трансплантации.

### 22.22 Урология

Извлекать: мочевые симптомы, простату, почки, мочевой пузырь, инфекции мочевых путей, УЗИ, анализ мочи, ПСА, оперативное лечение.

### 22.23 Хирургия

Извлекать: острый живот, хирургическую патологию, показания к операции, предоперационную подготовку, послеоперационное ведение, осложнения, госпитализацию.

### 22.24 Эндокринология, нарушения обмена веществ

Извлекать: сахарный диабет, гликемию, HbA1c, щитовидную железу, ТТГ, Т4, ожирение, метаболические нарушения, инсулинотерапию, сахароснижающую терапию.

---

## 23. Особые правила медицинского текста

Не терять единицы измерения:

```text
мг
мкг
г
мл
л
ммоль/л
мкмоль/л
мл/мин/1,73 м2
мм рт. ст.
%
раз в сутки
раза в неделю
```

Не терять дозировки:

```text
500 мг 2 раза в сутки
0,5 мг/кг
не более 40 мг/сут
в течение 7-10 дней
```

Не терять логические условия:

```text
при наличии
при отсутствии
если
в случае
по показаниям
при подозрении
после исключения
до получения результатов
```

Нормализовать списки, но сохранять исходный текст.

---

## 24. Тесты

Добавить обезличенные тестовые КЗ в:

```text
tests/fixtures/consultations/
```

### test_consultation_parser.py

Проверить:

1. Извлекается специальность врача.
2. Извлекается дата консультации.
3. Извлекается ФИО.
4. Извлекается дата рождения.
5. Рассчитывается возраст.
6. Извлекаются жалобы.
7. Извлекается анамнез.
8. Извлекается диагноз.
9. Извлекаются рекомендации.
10. Извлекаются обследования.

### test_diagnosis_parser.py

Проверить:

```text
K30. Диспепсия -> icd10_code=K30
L93.0. Дискоидная красная волчанка -> icd10_code=L93.0
I80.1. Флебит и тромбофлебит бедренной вены -> icd10_code=I80.1
Дискоидная красная волчанка ? -> certainty=suspected
Нельзя исключить инвазию -> certainty=suspected, safety_flag=possible_malignancy
undefined -> quality warning
```

### test_medication_parser.py

Проверить:

```text
Ривороксабан 20 мг раз в день постоянно
Гидроксихлорохин 200 мг по 1 таблетке 2 раза/сутки 2 недели
Преднизолон 5 мг по 12 таб в сутки
С 12.08.24 - Преднизолон 5 мг по 11 таб в сутки
Тримедат форте 1 т 2 раза в день 28 дней
```

### test_protocol_matcher.py

Проверить:

1. По `L93.0` подбирается дерматологический протокол.
2. По `I80.1` подбирается протокол по сосудистой патологии, если он есть в базе.
3. По `K30` подбирается гастроэнтерологический протокол.
4. Если нет МКБ, подбор идет по тексту диагноза.
5. Детский протокол не применяется к взрослому пациенту.
6. Протокол для беременных не применяется к пациенту без признаков беременности.

### test_compliance_engine.py

Проверить:

1. Обязательные обследования найдены в рекомендациях.
2. Выполненные обследования не считаются отсутствующими.
3. Диагноз со знаком вопроса оценивается как suspected.
4. Красный флаг переводит отчет в `manual_review_required`.
5. Отсутствие возраста снижает confidence, но не всегда снижает compliance.
6. Неприменимые по возрасту правила не участвуют в расчете балла.

---

## 25. Критерии приемки

Проект считается улучшенным, если:

1. Можно загрузить одно КЗ в PDF.
2. Из КЗ извлекаются дата, врач, специальность, пациент, дата рождения и возраст.
3. Из КЗ извлекаются жалобы, анамнез, объективный статус, обследования, диагнозы, назначения и рекомендации.
4. Приложение различает выполненные и рекомендованные обследования.
5. Приложение извлекает МКБ-коды.
6. Приложение распознает диагнозы со знаком вопроса как suspected.
7. Приложение распознает `undefined` как проблему качества.
8. Приложение извлекает лекарства, дозировки, кратность и длительность.
9. Приложение извлекает длинную схему снижения дозы как schedule.
10. Приложение рассчитывает возраст на дату консультации.
11. Приложение использует возраст при подборе протоколов и правил.
12. Приложение не применяет детские правила к взрослым пациентам.
13. Приложение не применяет правила для беременных к пациентам без признаков беременности.
14. Приложение подбирает протокол по МКБ и тексту диагноза.
15. Приложение анализирует несколько диагнозов в одном КЗ.
16. Приложение формирует JSON-отчет.
17. Приложение формирует Markdown-отчет.
18. Приложение дает разбивку оценки по блокам.
19. Приложение показывает missing required items.
20. Приложение показывает warnings.
21. Приложение показывает critical issues.
22. Приложение показывает source_refs на протоколы.
23. Приложение не выдумывает соответствие, если данных недостаточно.
24. При недостатке данных ставится `insufficient_data`.
25. При красных флагах ставится `manual_review_required`.
26. Обработка одного плохого КЗ не ломает batch-анализ.
27. Есть тесты на реальные обезличенные КЗ.
28. README описывает запуск анализа КЗ.
29. Есть аудит текущего проекта в `docs/current_project_audit.md`.
30. Существующая функциональность проекта не сломана.

---

## 26. Что не делать

Не нужно:

1. Ставить диагноз пациенту.
2. Давать медицинскую рекомендацию пациенту.
3. Исправлять лечение врача.
4. Считать назначение ошибкой без ссылки на протокол.
5. Применять протокол без проверки возраста и применимости.
6. Снижать оценку за правило, которое неприменимо к пациенту.
7. Игнорировать диагнозы со знаком вопроса.
8. Игнорировать красные флаги.
9. Делать вывод без source_refs.
10. Удалять существующий код проекта без аудита.
11. Подключать платные API без отдельного согласования.
12. Делать OCR всех документов на первом этапе.
13. Делать веб-интерфейс, если его нет в текущем проекте и это не согласовано.

---

## 27. Главный принцип

Каждый вывод приложения должен быть проверяемым:

```text
Что найдено в КЗ
Какой протокол применен
Почему протокол применим
Какое правило проверено
Что найдено или не найдено
Какой статус присвоен
На какой источник это опирается
```

Приложение должно быть не чат-ботом с мнением, а проверяемым инструментом экспертной оценки консультативных заключений по клиническим протоколам.

---

## 28. Рекомендуемая первая команда для Cursor

После добавления этого файла в проект написать в Cursor:

```text
Изучи файл docs/cursor_task_protocols_and_consultations.md.
Сначала не пиши код. Проведи аудит текущего проекта и составь план изменений.
После аудита создай docs/current_project_audit.md и docs/implementation_plan.md.
Затем начинай реализацию по этапам, не ломая существующую функциональность.
Если есть презентация в docs/project_presentation.pdf, изучи ее и учти текущее состояние проекта.
Используй data/examples/consultations как тестовые примеры реальных консультативных заключений.
```
