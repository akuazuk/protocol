# Задание для Cursor: улучшение проверки КЗ по протоколам РБ

## 1. Контекст

В проекте уже реализована сильная базовая архитектура:

- скачивание протоколов Минздрава РБ;
- разбор PDF;
- построение корпуса `chunks.jsonl`;
- карточки протоколов `protocol_cards.jsonl`;
- каталог клинических правил;
- RAG-поиск;
- structured analysis;
- `compliance_engine`;
- `safety_checker`;
- гибридный итоговый процент;
- CLI `python -m scripts.check_kz`;
- API/UI `POST /api/consult-review`.

Эту архитектуру не нужно переписывать с нуля. Нужно усилить существующую систему так, чтобы она стала проверяемым аудиторским инструментом оценки консультативных заключений на соответствие:

1. формальным требованиям к КЗ;
2. клиническим протоколам Минздрава РБ;
3. возрасту, полу, беременности, специальности врача, диагнозу и МКБ;
4. фактическому содержанию КЗ: жалобы, анамнез, объективный статус, обследования, назначения, контроль, маршрутизация.

Главная цель - не просто дать процент, а объяснить, из чего он получился, какие требования выполнены, какие не выполнены, какие правила неприменимы и где нужны ручная проверка или уточнение.

---

## 2. Главный принцип доработки

Не ломать текущие CLI, API, UI и существующую логику.

Нужно:

```text
1. Изучить текущую архитектуру.
2. Создать docs/improvement_plan.md.
3. Улучшать систему по этапам.
4. Добавлять regression-тесты на реальных обезличенных КЗ.
5. Сохранять совместимость с текущими env-переменными и пайплайном.
```

Запрещено:

```text
- переписывать проект с нуля;
- удалять рабочие модули без необходимости;
- заменять deterministic scoring на LLM-score;
- делать медицинские выводы без source_refs;
- снижать оценку за правила, которые неприменимы по возрасту/полу/беременности;
- применять детские протоколы к взрослым и наоборот;
- игнорировать диагнозы со знаком вопроса;
- игнорировать красные флаги.
```

---

## 3. Новая целевая логика оценки

Разделить оценку на 3 независимых слоя:

```text
A. Формальная полнота и качество КЗ
B. Применимость протокола
C. Соответствие КЗ требованиям протокола
```

Почему это важно:

- КЗ может быть хорошо оформлено, но не соответствовать протоколу.
- КЗ может соответствовать протоколу, но быть плохо оформлено.
- Протокол может быть найден по МКБ, но быть неприменимым по возрасту.
- Если протокол не найден, нельзя искусственно ставить низкий clinical score - нужно ставить `insufficient_protocol_data` или `low_confidence`.

---

## 4. Обновить score model

Добавить раздельные показатели:

```python
class ScoreBreakdown(BaseModel):
    documentation_score: float | None = None
    patient_data_score: float | None = None
    protocol_applicability_score: float | None = None
    diagnosis_score: float | None = None
    diagnostic_criteria_score: float | None = None
    required_exams_score: float | None = None
    treatment_score: float | None = None
    safety_score: float | None = None
    follow_up_score: float | None = None
    confidence_score: float | None = None
    overall_score: float | None = None
```

Важно:

```text
overall_score - насколько КЗ соответствует требованиям и протоколам
confidence_score - насколько системе хватает данных для надежной оценки
```

Если протокол не найден, КЗ плохо распарсено или правила отсутствуют, снижать `confidence_score`, а не выдумывать точный clinical score.

---

## 5. Обновить статусы

Использовать статусы:

```text
compliant
mostly_compliant
partially_compliant
non_compliant
insufficient_data
insufficient_protocol_data
low_confidence
manual_review_required
```

Правила:

```text
- Если данных КЗ недостаточно - insufficient_data.
- Если протокол не найден или нет правил по нозологии - insufficient_protocol_data.
- Если найден critical red flag без обработки - manual_review_required.
- Если confidence_score низкий - low_confidence, даже если overall_score рассчитан.
```

---

## 6. Усилить requirement_checker

Доработать `requirement_checker` и `config/kz_requirements.yaml`.

### 6.1 Required для любого КЗ

```yaml
required:
  consultation_date:
    title: "Дата консультации"
    severity_if_missing: major
    evidence_fields: ["consultation_date"]

  doctor_specialty:
    title: "Специальность врача"
    severity_if_missing: major
    evidence_fields: ["doctor_specialty"]

  patient_identity:
    title: "Данные пациента"
    severity_if_missing: major
    evidence_fields:
      - "patient.full_name"
      - "patient.birth_date"
      - "patient.age_years"

  diagnosis:
    title: "Диагноз"
    severity_if_missing: critical
    evidence_fields:
      - "diagnoses"
      - "sections.diagnosis_text"

  objective_status:
    title: "Объективный статус"
    severity_if_missing: major
    evidence_fields:
      - "sections.objective_status"

  recommendations:
    title: "Рекомендации"
    severity_if_missing: critical
    evidence_fields:
      - "sections.recommendations_exams"
      - "sections.recommendations_treatment"
      - "sections.general_recommendations"

  doctor_signature:
    title: "Идентификация врача"
    severity_if_missing: major
    evidence_fields:
      - "doctor_name"
      - "sections.doctor_signature"
```

### 6.2 Conditional

```yaml
conditional:
  complaints:
    title: "Жалобы"
    required_if:
      active_problem: true
      first_visit: true
    severity_if_missing: major

  anamnesis:
    title: "Анамнез"
    required_if:
      first_visit: true
      new_condition: true
    severity_if_missing: major

  icd10:
    title: "Код МКБ-10"
    required_if:
      diagnosis_present: true
    severity_if_missing: minor

  local_status:
    title: "Локальный статус"
    required_if:
      doctor_specialty_in:
        - "Дерматолог"
        - "Флеболог"
        - "Хирург"
        - "Травматолог"
        - "Ортопед"
        - "Офтальмолог"
        - "Оториноларинголог"
    severity_if_missing: major

  recommended_exams:
    title: "Дообследование"
    required_if:
      has_suspected_diagnosis: true
    severity_if_missing: major

  follow_up:
    title: "Повторная явка или контроль"
    required_if:
      has_treatment: true
      has_suspected_diagnosis: true
      has_red_flag: true
    severity_if_missing: major

  routing:
    title: "Маршрутизация"
    required_if:
      has_red_flag: true
    severity_if_missing: critical

  transportability:
    title: "Транспортабельность"
    required_if:
      patient_transfer_required: true
    severity_if_missing: critical
```

### 6.3 Recommended

```yaml
recommended:
  height:
    title: "Рост"
  weight:
    title: "Вес"
  bmi:
    title: "ИМТ"
  allergies:
    title: "Аллергоанамнез"
  current_medications:
    title: "Текущие препараты"
  risk_factors:
    title: "Факторы риска"
  symptom_duration:
    title: "Длительность симптомов"
  emergency_criteria:
    title: "Критерии срочного обращения"
```

---

## 7. Доработать consult_parser

Текущий парсер нужно усилить под реальные КЗ.

Он должен распознавать:

```text
- несколько диагнозов в одном КЗ;
- повторяющиеся блоки "Диагноз:";
- МКБ-коды;
- диагнозы без МКБ;
- диагнозы со знаком вопроса;
- фразы "нельзя исключить", "подозрение", "вероятно";
- слово "undefined" как дефект качества;
- шаблонные блоки ">>> L30 ...", ">>> L93.0 ...";
- блоки "ОБСЛЕДОВАНИЯ ОБЯЗАТЕЛЬНЫЕ";
- блоки "ОБСЛЕДОВАНИЯ ДОПОЛНИТЕЛЬНЫЕ";
- блоки "ПОВТОРНЫЙ ПРИЕМ";
- уже выполненные обследования;
- рекомендованные обследования;
- контрольные обследования;
- препараты;
- дозы;
- кратность;
- длительность;
- длинные схемы снижения дозы по датам.
```

### 7.1 Diagnosis objects

Не хранить один общий диагноз строкой. Нужно хранить список:

```python
class ConsultationDiagnosis(BaseModel):
    diagnosis_id: str
    raw_text: str
    icd10_code: str | None = None
    diagnosis_name: str | None = None

    role: Literal[
        "primary",
        "secondary",
        "comorbidity",
        "symptom",
        "finding",
        "red_flag_finding",
        "unknown"
    ] = "unknown"

    certainty: Literal[
        "confirmed",
        "suspected",
        "excluded",
        "unclear"
    ] = "unclear"

    safety_flags: list[str] = []
    source_section: str | None = None
    source_text: str | None = None
```

Правила:

```text
- "?" -> suspected
- "нельзя исключить" -> suspected
- "подозрение" -> suspected
- "опухолевое образование" -> red_flag_finding + possible_malignancy
- R-коды и Z-коды не должны вытеснять нозологические диагнозы
- симптом или синдром должен быть role=symptom, если рядом есть нозологический диагноз
```

### 7.2 TemplateBlock

Добавить поддержку шаблонных блоков:

```python
class TemplateBlock(BaseModel):
    block_diagnosis_text: str
    icd10_code: str | None = None
    block_type: Literal[
        "required_exams",
        "additional_exams",
        "follow_up",
        "treatment",
        "care",
        "unknown"
    ]
    items: list[str]
    source_text: str
```

Пример распознавания:

```text
>>> L30 Экзема кожи ?:
* ОБСЛЕДОВАНИЯ ОБЯЗАТЕЛЬНЫЕ:
- Общий анализ крови
- Общий анализ мочи
- Исследование на сифилис
```

Должно стать:

```json
{
  "icd10_code": "L30",
  "block_type": "required_exams",
  "items": [
    "Общий анализ крови",
    "Общий анализ мочи",
    "Исследование на сифилис"
  ]
}
```

### 7.3 Medication schedule

Длинные схемы снижения дозы не склеивать в одну строку. Хранить как:

```python
class MedicationScheduleStep(BaseModel):
    start_date: date | None = None
    end_date: date | None = None
    dose_text: str
    frequency_text: str | None = None
    daily_dose_text: str | None = None
```

Примеры, которые должны парситься:

```text
Преднизолон 5 мг по 12 таб в сутки
С 12.08.24 - Преднизолон 5 мг по 11 таб в сутки
С 19.08.24 - Преднизолон 5 мг по 10 таб в сутки
С 26.08.24 - Преднизолон 5 мг по 9 таб в сутки
```

---

## 8. Доработать ProtocolMatcher

Текущий `match_protocol_cards` нужно усилить.

### 8.1 Ранжирование

Рекомендуемая формула:

```text
match_score =
  40% МКБ
  20% текст диагноза
  15% специальность врача / рубрика
  10% возраст и population протокола
  5% пол / беременность
  5% обследования
  5% жалобы
```

### 8.2 Жесткие ограничения

```text
- детский протокол не применять к взрослому;
- взрослый протокол не применять к ребенку;
- протокол для беременных не применять без признаков беременности;
- протокол по другой специальности не применять без сильного совпадения МКБ;
- симптом-коды R/Z не должны вытеснять нозологические коды;
- диагноз suspected должен проверяться по правилам для предварительного диагноза, а не как confirmed.
```

### 8.3 MatchResult

```python
class ProtocolMatchResult(BaseModel):
    diagnosis_id: str
    protocol_id: str
    protocol_title: str
    rubric_name: str
    matched_condition: str | None = None

    match_score: float
    applicability: Literal[
        "applicable",
        "possibly_applicable",
        "not_applicable",
        "unknown"
    ]

    match_reasons: list[str] = []
    mismatch_reasons: list[str] = []

    age_applicability: str | None = None
    sex_applicability: str | None = None
    pregnancy_applicability: str | None = None

    source_refs: list[dict] = []
```

---

## 9. Rule model 2.0

Расширить модель клинического правила.

```python
class ProtocolRule(BaseModel):
    rule_id: str
    protocol_id: str
    condition_name: str | None = None
    icd10_codes: list[str] = []

    rule_type: Literal[
        "diagnosis_structure_rule",
        "clinical_criterion_rule",
        "diagnostic_criterion_rule",
        "required_exam_rule",
        "conditional_exam_rule",
        "performed_or_recommended_exam_rule",
        "treatment_group_rule",
        "drug_rule",
        "drug_dose_rule",
        "drug_duration_rule",
        "non_drug_rule",
        "follow_up_rule",
        "routing_rule",
        "red_flag_rule",
        "contraindication_rule",
        "age_applicability_rule",
        "sex_applicability_rule",
        "pregnancy_applicability_rule",
        "informational_rule"
    ]

    severity: Literal[
        "required",
        "conditional",
        "recommended",
        "warning",
        "forbidden",
        "informational"
    ]

    applicability: RuleApplicability
    evidence_targets: list[str]
    criteria: list[dict] = []
    expected_items: list[str] = []
    forbidden_items: list[str] = []

    source: SourceRef
```

```python
class RuleApplicability(BaseModel):
    age_groups: list[str] = []
    age_min_years: int | None = None
    age_max_years: int | None = None
    sex: Literal["male", "female", "any", "unknown"] = "unknown"
    pregnancy: Literal["required", "excluded", "any", "unknown"] = "unknown"
    condition_certainty: list[Literal["confirmed", "suspected", "unclear"]] = []
    care_setting: list[str] = []
```

`evidence_targets`:

```text
complaints
anamnesis
objective_status
local_status
performed_exams
recommended_exams
diagnosis
treatment
medications
follow_up
routing
allergies
comorbidities
```

---

## 10. Доработать извлечение правил из протоколов

Сейчас есть типы правил `diagnosis_formula`, `diagnostic_criterion`, `required_exam`, `keyword_presence`, `population_mismatch`.

Нужно добавить:

```text
diagnosis_structure_rule
clinical_criterion_rule
diagnostic_criterion_rule
required_exam_rule
conditional_exam_rule
performed_or_recommended_exam_rule
treatment_group_rule
drug_rule
drug_dose_rule
drug_duration_rule
non_drug_rule
follow_up_rule
routing_rule
red_flag_rule
contraindication_rule
age_applicability_rule
sex_applicability_rule
pregnancy_applicability_rule
```

Важно:

```text
- Не выдумывать правила.
- Каждое правило должно иметь source_ref.
- Если правило невозможно формализовать, сохранять как informational_rule.
- Не использовать только keyword_presence для лечения.
- Для лечения пытаться извлекать препарат/группу, дозу, кратность, длительность, условия назначения.
```

---

## 11. Добавить table_rule_extractor

Создать модуль:

```text
clinical_knowledge/table_rule_extractor.py
```

Задача - превращать таблицы протоколов в правила.

Таблицы часто содержат:

```text
- обязательные обследования;
- дополнительные обследования;
- схемы лечения;
- дозировки;
- длительность терапии;
- критерии тяжести;
- критерии госпитализации;
- сроки контроля.
```

Модуль должен:

```text
1. Читать table_block чанки.
2. Определять тип таблицы.
3. Разбирать строки таблицы.
4. Создавать ProtocolRule.
5. Сохранять source_ref: документ, страница, таблица, строка.
```

Пример результата:

```json
{
  "rule_type": "required_exam_rule",
  "severity": "required",
  "expected_items": ["общий анализ крови", "общий анализ мочи"],
  "evidence_targets": ["performed_exams", "recommended_exams"],
  "source": {
    "protocol_id": "...",
    "page": 12,
    "section_title": "Диагностика",
    "table_index": 2,
    "row_index": 4,
    "quote": "..."
  }
}
```

---

## 12. Evidence map

Добавить отдельную карту доказательств для каждого проверенного правила.

```python
class EvidenceMapItem(BaseModel):
    rule_id: str
    rule_type: str
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

Примеры решений:

```text
satisfied - требование выполнено
satisfied_by_recommendation - обследование не выполнено, но назначено
missing - отсутствует и не назначено
not_applicable - правило неприменимо
manual_review - требуется ручная проверка
unknown - не удалось оценить
```

Evidence map должен попадать в JSON и Markdown-отчет.

---

## 13. Доработать compliance_engine

### 13.1 Обследования

Новая логика:

```text
present_performed - обследование уже выполнено
present_recommended - обследование назначено
missing_required - не выполнено и не назначено
missing_conditional - не выполнено при наличии показаний
not_applicable - правило неприменимо
unknown - не удалось оценить
```

Правило:

```text
Если обязательное обследование уже выполнено - не снижать оценку.
Если обязательное обследование не выполнено, но назначено - считать частично выполненным или satisfied_by_recommendation.
Если обязательное обследование не выполнено и не назначено - missing_required.
```

### 13.2 Диагноз

```text
confirmed diagnosis - требует подтверждающих критериев;
suspected diagnosis - требует дообследования;
finding/red_flag_finding - требует маршрутизации или контроля;
symptom diagnosis - не должен вытеснять нозологический диагноз.
```

### 13.3 Лечение

Проверять:

```text
- препарат или группу;
- дозу;
- кратность;
- длительность;
- возрастные ограничения;
- противопоказания;
- контроль безопасности;
- немедикаментозные рекомендации.
```

Не снижать оценку за drug_dose_rule, если в протоколе нет формализованной дозировки или правило не извлечено с достаточной уверенностью. В таком случае ставить `unknown` или `manual_review`.

### 13.4 Неприменимые правила

```text
Неприменимое по возрасту/полу/беременности/статусу диагноза правило не должно снижать оценку.
```

---

## 14. Усилить SafetyChecker

Добавить классы красных флагов в `config/red_flags.yaml`.

```yaml
possible_malignancy:
  keywords:
    - опухолевое образование
    - нельзя исключить инвазию
    - подозрение на злокачественное
    - новообразование
  required_actions:
    - дообследование
    - профильная консультация
    - маршрутизация
    - контроль срока
  cap_if_unhandled: 45

thrombosis:
  keywords:
    - флеботромбоз
    - тромбоз глубоких вен
    - ТГВ
    - тромбофлебит
  required_actions:
    - антикоагулянтная терапия
    - контроль УЗИ
    - повторная консультация
  cap_if_unhandled: 55

systemic_autoimmune:
  keywords:
    - дискоидная красная волчанка
    - системная красная волчанка
    - ANA
    - anti-DNA
  required_actions:
    - лабораторное обследование
    - оценка системности
    - фотозащита
    - контроль
  cap_if_unhandled: 65

drug_safety:
  keywords:
    - преднизолон
    - гидроксихлорохин
    - ривароксабан
    - ривороксабан
  required_actions:
    - длительность
    - контроль безопасности
    - повторная явка
```

Логика:

```text
Если red flag найден и required_actions присутствуют в КЗ - status=handled или partially_handled.
Если red flag найден и required_actions отсутствуют - status=not_handled, manual_review_required.
Если critical red flag не обработан - применить safety cap.
```

---

## 15. Зафиксировать роль LLM

Текущая архитектура использует Gemini для фокуса, RAG и экспертной оценки. Это можно оставить.

Но нужно закрепить правило:

```text
LLM не имеет права повышать итоговый процент.
```

LLM может:

```text
- формировать экспертное объяснение;
- помогать извлекать спорные факты;
- суммаризировать evidence;
- формировать narrative report.
```

Deterministic engine должен считать:

```text
- итоговый процент;
- score breakdown;
- critical issues;
- safety cap;
- manual_review_required;
- insufficient_data.
```

Добавить в JSON-отчет:

```json
{
  "score_source": "deterministic",
  "llm_used_for": [
    "query_focus",
    "evidence_summarization",
    "expert_explanation"
  ],
  "llm_score_ignored": true
}
```

---

## 16. Regression-тесты на реальных КЗ

Добавить обезличенные fixture-файлы в:

```text
tests/fixtures/consultations/
```

Минимальный набор тестов:

### 16.1 gastro_1

Проверить:

```text
- возраст рассчитывается на дату консультации;
- извлекаются K30, Q43.8, E61.1, E80.4, R14, K82.8;
- "опухолевое образование сигмовидной кишки" -> possible_malignancy;
- "нельзя исключить инвазию" -> suspected + critical red flag;
- колоноскопия под седацией -> recommended_exam;
- ОАК/ЭКГ перед седацией -> pre_procedure_exams;
- уже выполненные ОАК, БАК, ФКС, ЭГДС, УЗИ, КТ не считаются missing.
```

### 16.2 mg_1

Проверить:

```text
- "undefined" -> data_quality issue;
- J06/J06.8 извлекаются;
- терапевт -> соответствующая рубрика/протокол;
- рекомендации извлекаются;
- неполный follow-up -> warning, не critical.
```

### 16.3 pl_1_d

Проверить:

```text
- L30 и L93.0 извлекаются отдельно;
- обязательные обследования распознаются как recommended required exams;
- дополнительные обследования отделяются от обязательных;
- повторный прием через 3-5 и 10-15 дней распознается;
- шаблонные блоки >>> разбираются как TemplateBlock.
```

### 16.4 pl_1_f

Проверить:

```text
- I80.1 извлекается;
- флеботромбоз -> thrombosis red flag;
- ривароксабан/ривороксабан 20 мг 1 раз в день постоянно -> medication;
- контроль УЗИ через 3 месяца -> handled safety action;
- повторная консультация флеболога -> follow_up.
```

### 16.5 pl_2_d_s

Проверить:

```text
- "Дискоидная красная волчанка ?" -> suspected;
- ANA/anti-DNA извлекаются как recommended_exams;
- гидроксихлорохин 200 мг 2 раза/сутки 2 недели -> medication;
- фотозащита -> non_drug_recommendation.
```

### 16.6 pl_2_d_s_2

Проверить:

```text
- длинная схема преднизолона извлекается как schedule;
- гидроксихлорохин меняет кратность с 2 раз/сутки на 1 раз/сутки;
- повторный осмотр после 11 недель -> follow_up;
- преднизолон -> drug_safety flag;
- есть план снижения дозы -> handled или partially_handled.
```

### 16.7 pl_2_d_s_3

Проверить:

```text
- повторная дерматологическая консультация объединяется в episode_of_care;
- динамика "новые очаги не появляются" распознается как improvement;
- терапия продолжается, но должна быть связана с предыдущим назначением.
```

---

## 17. Обновить Markdown-отчет

Отчет должен включать:

```text
1. Итоговый статус.
2. overall_score.
3. confidence_score.
4. score breakdown.
5. Что распарсено из КЗ.
6. Какие протоколы подобраны и почему.
7. Какие протоколы не применены и почему.
8. Какие правила неприменимы по возрасту/полу/беременности.
9. Evidence map.
10. Выполненные требования.
11. Частично выполненные требования.
12. Отсутствующие обязательные требования.
13. Critical issues.
14. Major issues.
15. Warnings.
16. Red flags.
17. Safety cap, если применен.
18. Source refs к КЗ.
19. Source refs к протоколам.
20. Ограничения оценки.
```

Пример блока:

```markdown
## Применимость протоколов

| Диагноз | МКБ | Протокол | Применимость | Причина |
|---|---|---|---|---|
| Диспепсия | K30 | ... | applicable | совпал МКБ, взрослый пациент, гастроэнтерология |
| Дискоидная красная волчанка | L93.0 | ... | applicable | совпал МКБ, дерматология, взрослый пациент |
| Детский протокол ... | L93.0 | not_applicable | возраст пациента 48 лет |
```

---

## 18. Обновить JSON-отчет

Добавить поля:

```json
{
  "overall_score": null,
  "confidence_score": null,
  "overall_status": "",
  "score_source": "deterministic",
  "llm_score_ignored": true,
  "score_breakdown": {},
  "protocol_matches": [],
  "not_applicable_protocols": [],
  "evidence_map": [],
  "issues": [],
  "critical_issues": [],
  "major_issues": [],
  "warnings": [],
  "safety_cap": {
    "applied": false,
    "reason": null,
    "cap_value": null
  },
  "limitations": []
}
```

---

## 19. Обновить batch summary

Добавить колонки:

```text
file
consultation_date
doctor_specialty
patient_age
patient_sex
diagnoses
icd10_codes
matched_protocols
not_applicable_protocols
overall_score
confidence_score
overall_status
critical_issues_count
major_issues_count
warnings_count
red_flags
safety_cap_applied
manual_review_required
insufficient_protocol_data
```

---

## 20. Приоритет реализации

Реализовать по этапам.

### Этап 1 - Improvement plan

Создать:

```text
docs/improvement_plan.md
```

В нем указать:

```text
- какие модули уже есть;
- какие будут расширены;
- какие новые файлы будут добавлены;
- какие тесты будут добавлены;
- какие риски есть;
- как не сломать текущий API/UI.
```

### Этап 2 - Parser + Evidence map

```text
- улучшить consult_parser;
- добавить TemplateBlock;
- улучшить diagnosis parser;
- улучшить medication schedule parser;
- добавить EvidenceMapItem.
```

### Этап 3 - Requirement checker

```text
- обновить kz_requirements.yaml;
- добавить required/conditional/recommended;
- добавить data_quality issues.
```

### Этап 4 - Protocol matcher

```text
- усилить match_score;
- добавить жесткие applicability rules;
- анализировать каждый диагноз отдельно.
```

### Этап 5 - Rule model 2.0

```text
- расширить ProtocolRule;
- добавить applicability;
- добавить evidence_targets;
- обновить rule checker.
```

### Этап 6 - Table rule extractor

```text
- извлекать правила из таблиц;
- сохранять source_ref строки таблицы.
```

### Этап 7 - Compliance engine

```text
- новая логика performed/recommended/missing;
- suspected/confirmed diagnosis logic;
- неприменимые правила не снижают score.
```

### Этап 8 - Safety checker

```text
- расширить red_flags.yaml;
- добавить handled/partially_handled/not_handled;
- добавить safety cap.
```

### Этап 9 - Scoring + confidence

```text
- добавить confidence_score;
- обновить статусы;
- зафиксировать deterministic score.
```

### Этап 10 - Reports + tests

```text
- обновить JSON;
- обновить Markdown;
- обновить batch summary;
- добавить regression tests.
```

---

## 21. Критерии приемки

Доработка считается выполненной, если:

```text
1. Текущие CLI/API/UI не сломаны.
2. Создан docs/improvement_plan.md.
3. Добавлен confidence_score.
4. Итоговый score считается deterministic engine.
5. LLM не повышает итоговый score.
6. Есть отдельный documentation_score.
7. Есть отдельный protocol_applicability_score.
8. Есть отдельный diagnosis_score.
9. Есть отдельный required_exams_score.
10. Есть отдельный treatment_score.
11. Парсер извлекает несколько диагнозов.
12. Парсер распознает МКБ.
13. Парсер распознает suspected диагнозы.
14. Парсер распознает "undefined" как data_quality issue.
15. Парсер распознает шаблонные блоки >>>.
16. Парсер разделяет обязательные и дополнительные обследования.
17. Парсер разделяет performed_exams и recommended_exams.
18. Парсер извлекает длинную схему преднизолона как schedule.
19. ProtocolMatcher учитывает возраст, пол, беременность и специальность.
20. Детский протокол не применяется к взрослому.
21. Взрослый протокол не применяется к ребенку.
22. Неприменимые правила не снижают оценку.
23. Обязательное обследование, назначенное врачом, не считается полностью missing.
24. Есть EvidenceMapItem для проверенных правил.
25. Red flags классифицируются по типам.
26. Critical red flag без обработки переводит отчет в manual_review_required.
27. Safety cap применяется и отображается в отчете.
28. Markdown-отчет показывает confidence и evidence map.
29. JSON-отчет содержит score_source, confidence_score, evidence_map, safety_cap.
30. Batch summary содержит confidence_score и manual_review_required.
31. Добавлены regression-тесты на обезличенных КЗ.
32. Все тесты проходят.
```

---

## 22. Первая команда для Cursor

После добавления этого файла в проект выполнить:

```text
Изучи docs/cursor_task_improve_kz_protocol_compliance.md.
Не переписывай проект с нуля.
Сначала создай docs/improvement_plan.md с анализом текущей архитектуры и планом безопасных изменений.
Затем реализуй доработки по этапам:
1. parser + evidence map;
2. requirement_checker;
3. protocol_matcher;
4. rule model 2.0;
5. table_rule_extractor;
6. compliance_engine;
7. safety_checker;
8. scoring + confidence;
9. reports;
10. regression tests.
Сохрани совместимость с текущими CLI, API и UI.
