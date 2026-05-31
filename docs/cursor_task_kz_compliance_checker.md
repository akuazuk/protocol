# ТЗ для Cursor: модуль проверки консультативных заключений на соответствие требованиям РБ и клиническим протоколам

## 1. Цель задачи

Необходимо разработать или доработать модуль приложения, который проверяет консультативные заключения врачей на соответствие:

1. Формальным требованиям к консультативному заключению в Республике Беларусь.
2. Клиническим протоколам Минздрава Республики Беларусь.
3. Внутреннему стандарту качества заполнения КЗ.
4. Возрасту, полу, беременности, специальности врача, диагнозу, МКБ-коду и другим параметрам пациента.

Модуль должен принимать консультативное заключение в PDF/DOCX/TXT/JSON, извлекать из него структурированные данные, подбирать релевантные протоколы и формировать отчет с оценкой соответствия.

Приложение не должно ставить диагноз, заменять врача или давать самостоятельные медицинские рекомендации. Оно должно проверять качество документа и соответствие КЗ протоколам.

---

## 2. Главный принцип работы

Каждая оценка должна быть объяснимой:

```text
Что найдено в КЗ
Какое требование проверяется
Какой протокол применен
Почему протокол применим
Что соответствует
Что отсутствует
Что требует ручной проверки
На какой источник опирается вывод
```

Запрещено делать вывод без ссылки на:

- фрагмент консультативного заключения;
- нормативное/внутреннее требование;
- пункт клинического протокола, если проверка связана с протоколом.

---

## 3. Что должен проверять модуль

Модуль должен выполнять два типа проверки:

```text
1. Проверка структуры и качества заполнения КЗ
2. Проверка соответствия КЗ клиническим протоколам
```

### 3.1 Проверка структуры и качества КЗ

Проверить наличие и корректность рубрик:

1. Шапка документа.
2. Дата и время консультации.
3. Медицинская организация.
4. Специальность врача.
5. Данные врача.
6. Данные пациента.
7. Возраст пациента на дату консультации.
8. Пол пациента.
9. Цель консультации.
10. Жалобы.
11. Анамнез заболевания.
12. Анамнез жизни и факторы риска.
13. Аллергоанамнез.
14. Лекарственный анамнез.
15. Объективный статус.
16. Локальный статус.
17. Данные выполненных обследований.
18. Диагноз.
19. Код МКБ-10.
20. Клиническое обоснование диагноза.
21. Рекомендации по обследованию.
22. Рекомендации по лечению.
23. Немедикаментозные рекомендации.
24. Маршрутизация.
25. Повторная явка/контроль.
26. Информированное согласие/отказы, если применимо.
27. Подпись и идентификация врача.

### 3.2 Проверка соответствия протоколам

Проверить:

1. Подобран ли правильный протокол.
2. Применим ли протокол по возрасту.
3. Применим ли протокол по полу.
4. Применим ли протокол по беременности.
5. Применим ли протокол по специальности врача.
6. Применим ли протокол по диагнозу и МКБ.
7. Есть ли обязательные клинические критерии.
8. Есть ли обязательные диагностические критерии.
9. Выполнены или назначены обязательные обследования.
10. Выполнены или назначены дополнительные обследования по показаниям.
11. Корректно ли сформулирован диагноз.
12. Соответствует ли лечение протоколу.
13. Соответствуют ли дозы, кратность и длительность, если они формализованы в протоколе.
14. Есть ли противопоказания.
15. Есть ли красные флаги.
16. Есть ли маршрутизация при красных флагах.
17. Есть ли повторная явка или контроль, если они нужны.

---

## 4. Архитектура модуля

Добавить или доработать папку:

```text
src/consultation_compliance/
  __init__.py
  models.py
  kz_parser.py
  kz_section_extractor.py
  patient_resolver.py
  diagnosis_extractor.py
  exam_extractor.py
  medication_extractor.py
  protocol_matcher.py
  requirement_checker.py
  protocol_compliance_checker.py
  scoring.py
  safety_checker.py
  report_builder.py
  batch_runner.py
```

Если в проекте уже есть похожие модули, не создавать дубли, а расширить существующую архитектуру.

---

## 5. Входные данные

Модуль должен принимать:

```text
PDF
DOCX
TXT
JSON
```

Примеры команд:

```bash
python -m src.main check-kz --file data/examples/consultations/gastro_1.pdf
python -m src.main check-kz --file data/examples/consultations/pl_1_d.pdf
python -m src.main check-kz-folder --folder data/examples/consultations
```

---

## 6. Выходные данные

Для каждого КЗ сформировать:

```text
data/reports/kz_checks/{consultation_id}.json
data/reports/kz_checks/{consultation_id}.md
```

Также для папки КЗ сформировать общий отчет:

```text
data/reports/kz_checks/batch_summary.csv
data/reports/kz_checks/batch_summary.md
```

---

## 7. Модели данных

Создать или доработать `src/consultation_compliance/models.py`.

### 7.1 ConsultationDocument

```python
class ConsultationDocument(BaseModel):
    consultation_id: str
    source_file: str
    source_file_type: str
    raw_text: str

    clinic_name: str | None = None

    consultation_date: date | None = None
    consultation_time: str | None = None

    doctor_specialty: str | None = None
    doctor_name: str | None = None
    doctor_position: str | None = None
    doctor_category: str | None = None

    patient: PatientContext
    sections: ConsultationSections

    diagnoses: list[ConsultationDiagnosis] = []
    performed_exams: list[ExamItem] = []
    recommended_exams: list[ExamItem] = []
    medications: list[MedicationItem] = []
    follow_up: list[FollowUpItem] = []

    extraction_quality: ExtractionQuality
```

### 7.2 PatientContext

```python
class PatientContext(BaseModel):
    full_name: str | None = None
    birth_date: date | None = None
    age_years: int | None = None
    age_months: int | None = None

    sex: Literal["male", "female", "unknown"] = "unknown"

    age_group: Literal[
        "newborn",
        "infant",
        "child",
        "adult",
        "elderly",
        "unknown"
    ] = "unknown"

    pregnancy: bool | None = None

    height_cm: float | None = None
    weight_kg: float | None = None
    bmi: float | None = None

    allergies: list[str] = []
    current_medications: list[str] = []
    comorbidities: list[str] = []
    surgeries: list[str] = []
    risk_factors: list[str] = []
```

Возраст рассчитывать на дату консультации.

Правила возрастных групп:

```text
newborn - 0-28 дней
infant - 29 дней - 1 год
child - 1-17 лет
adult - 18-64 года
elderly - 65 лет и старше
unknown - если возраст определить невозможно
```

### 7.3 ConsultationSections

```python
class ConsultationSections(BaseModel):
    header: str | None = None
    consent_text: str | None = None
    complaints: str | None = None
    anamnesis: str | None = None
    life_history: str | None = None
    allergy_history: str | None = None
    medication_history: str | None = None
    surgical_history: str | None = None
    objective_status: str | None = None
    local_status: str | None = None
    exam_results: str | None = None
    diagnosis_text: str | None = None
    recommendations_exams: str | None = None
    recommendations_treatment: str | None = None
    non_drug_recommendations: str | None = None
    routing: str | None = None
    follow_up_text: str | None = None
    doctor_signature: str | None = None
```

### 7.4 ConsultationDiagnosis

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
        "finding",
        "unknown"
    ] = "unknown"

    certainty: Literal[
        "confirmed",
        "suspected",
        "excluded",
        "unclear"
    ] = "unclear"

    source_text: str | None = None
```

Правила:

```text
Если есть "?", "подозрение", "нельзя исключить", "вероятно", "под вопросом" - certainty = suspected
Если есть "основной" - diagnosis_role = primary
Если диагноз похож на симптом - diagnosis_role = symptom
Если это результат обследования, а не диагноз - diagnosis_role = finding
```

### 7.5 ExamItem

```python
class ExamItem(BaseModel):
    exam_id: str
    raw_text: str
    exam_name: str | None = None

    exam_type: Literal[
        "laboratory",
        "instrumental",
        "imaging",
        "functional",
        "consultation",
        "unknown"
    ] = "unknown"

    status: Literal[
        "performed",
        "recommended",
        "planned",
        "control",
        "unknown"
    ] = "unknown"

    exam_date: date | None = None
    result_text: str | None = None
    abnormal_flag: bool | None = None
```

### 7.6 MedicationItem

```python
class MedicationItem(BaseModel):
    medication_id: str
    raw_text: str

    drug_name: str | None = None
    active_substance: str | None = None

    dose_text: str | None = None
    dose_value: float | None = None
    dose_unit: str | None = None

    frequency_text: str | None = None
    duration_text: str | None = None
    route: str | None = None

    start_date: date | None = None
    end_date: date | None = None

    schedule: list[MedicationScheduleStep] = []
```

### 7.7 MedicationScheduleStep

```python
class MedicationScheduleStep(BaseModel):
    start_date: date | None = None
    end_date: date | None = None
    dose_text: str
    frequency_text: str | None = None
    daily_dose_text: str | None = None
```

### 7.8 ComplianceIssue

```python
class ComplianceIssue(BaseModel):
    issue_id: str

    category: Literal[
        "missing_required_section",
        "missing_conditional_section",
        "data_quality",
        "protocol_mismatch",
        "diagnosis_issue",
        "exam_issue",
        "treatment_issue",
        "safety_issue",
        "routing_issue",
        "age_applicability_issue",
        "manual_review"
    ]

    severity: Literal[
        "info",
        "minor",
        "major",
        "critical"
    ]

    title: str
    description: str

    consultation_evidence: list[str] = []
    protocol_evidence: list[str] = []
    source_refs: list[dict] = []

    affects_score: bool = True
```

### 7.9 KzComplianceReport

```python
class KzComplianceReport(BaseModel):
    consultation_id: str
    source_file: str

    patient_summary: dict
    doctor_summary: dict

    overall_score: float | None
    overall_status: Literal[
        "compliant",
        "mostly_compliant",
        "partially_compliant",
        "non_compliant",
        "insufficient_data",
        "manual_review_required"
    ]

    score_breakdown: ScoreBreakdown

    structural_assessment: StructuralAssessment
    protocol_assessment: ProtocolAssessment
    diagnosis_assessments: list[DiagnosisAssessment]
    exam_assessments: list[ExamAssessment]
    treatment_assessments: list[TreatmentAssessment]
    safety_assessments: list[SafetyAssessment]

    issues: list[ComplianceIssue]
    critical_issues: list[ComplianceIssue]
    warnings: list[ComplianceIssue]

    explanation: str
```

---

## 8. Парсинг КЗ

Создать `kz_parser.py`.

Модуль должен извлекать из КЗ:

```text
медицинскую организацию
дату и время консультации
специальность врача
ФИО врача
категорию врача
ФИО пациента
дату рождения
возраст
пол
жалобы
анамнез
аллергоанамнез
лекарственный анамнез
хирургический анамнез
объективный статус
локальный статус
данные обследований
диагнозы
МКБ-коды
рекомендации по обследованию
рекомендации по лечению
немедикаментозные рекомендации
лекарственные назначения
дозировки
кратность
длительность
схемы снижения дозы
дату повторной явки
подпись врача
```

Распознавать заголовки:

```text
Жалобы:
Анамнез:
Aнамнез:
Аллергия на лс:
Аллергоанамнез:
Лекарственный:
Хирургические вмешательства:
Объективный статус:
Локальный статус:
Данные обследований:
Диагноз:
Рекомендации:
Рекомендации по обследованию:
Рекомендации по лечению:
Дата повторной явки:
Врач:
```

Учитывать ошибки распознавания:

```text
Aнамнез вместо Анамнез
лс вместо ЛС
теч вместо течение
учавствует вместо участвует
гастропатяи вместо гастропатия
```

---

## 9. Проверка обязательных рубрик КЗ

Создать `requirement_checker.py`.

Разделить требования на:

```text
required
conditional
recommended
not_applicable
```

### 9.1 Required для любого КЗ

```text
Дата консультации
Специальность врача
Данные пациента
Дата рождения или возраст
Диагноз
Объективный статус
Рекомендации
ФИО/подпись врача
```

Если отсутствует любое из required-полей, создать issue:

```text
category = missing_required_section
severity = major
```

Если отсутствуют диагноз или рекомендации:

```text
severity = critical
```

### 9.2 Conditional

```text
Жалобы - обязательны, если пациент обращается с активной проблемой
Анамнез - обязателен при первичном приеме или новом заболевании
Данные обследований - обязательны, если диагноз подтверждается исследованиями
МКБ-10 - обязателен, если формат организации/МИС предусматривает кодирование
Локальный статус - обязателен для дерматологии, хирургии, травматологии, флебологии, офтальмологии, ЛОР и других профильных осмотров
Рекомендации по обследованию - обязательны при предварительном диагнозе или недостатке данных
Дата повторного приема - обязательна, если назначен контроль
Маршрутизация - обязательна при красных флагах
Транспортабельность - обязательна при переводе пациента
```

### 9.3 Recommended

```text
Рост
Вес
ИМТ
Факторы риска
Сопутствующие заболевания
Текущие препараты
Аллергии
Длительность симптомов
Критерии эффективности лечения
Критерии срочного обращения
```

---

## 10. Проверка качества данных КЗ

Создать проверки:

```text
Нет даты консультации
Нет возраста или даты рождения
Нет пола
Нет специальности врача
Нет жалоб при первичном приеме
Нет анамнеза
Нет объективного статуса
Диагноз без МКБ
МКБ не соответствует тексту диагноза
Диагноз со знаком вопроса, но нет дообследования
Есть красный флаг, но нет маршрутизации
Назначен препарат без дозировки
Назначен препарат без кратности
Назначен препарат без длительности
Есть слово undefined
Повторный прием указан без срока
Обследования перечислены без дат
Результаты обследований указаны без значений при значимых отклонениях
Нет подписи/идентификации врача
```

Особое правило:

```text
Если найдено слово undefined - это всегда data_quality issue.
```

---

## 11. Подбор протокола

Создать `protocol_matcher.py`.

Подбор протокола должен учитывать:

```text
МКБ-код
текст диагноза
специальность врача
возраст пациента
пол пациента
беременность
жалобы
обследования
дату консультации
дату действия протокола
рубрику протокола
```

### 11.1 ProtocolMatchResult

```python
class ProtocolMatchResult(BaseModel):
    protocol_id: str
    protocol_title: str
    rubric_name: str
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

    source_refs: list[dict]
```

### 11.2 Правила применимости

```text
Если совпал МКБ, но возраст не подходит - протокол не применять автоматически.
Если протокол детский, а пациент взрослый - not_applicable.
Если протокол взрослый, а пациент ребенок - not_applicable.
Если протокол для беременных, а беременность не указана - possibly_applicable или not_applicable.
Если диагноз со знаком вопроса - применять как suspected, а не как confirmed.
Если диагноз без МКБ - подбирать по тексту диагноза и специальности.
Если несколько диагнозов - анализировать каждый отдельно.
```

---

## 12. Проверка соответствия протоколу

Создать `protocol_compliance_checker.py`.

Проверять каждый применимый протокол по блокам:

```text
diagnosis
clinical_criteria
diagnostic_criteria
required_exams
conditional_exams
treatment
drug_treatment
non_drug_treatment
follow_up
hospitalization
contraindications
red_flags
```

### 12.1 Диагноз

Проверить:

```text
Есть ли диагноз
Есть ли МКБ
Соответствует ли МКБ тексту диагноза
Есть ли обязательные компоненты формулировки диагноза
Есть ли форма
Есть ли стадия
Есть ли степень тяжести
Есть ли фаза
Есть ли осложнения
Есть ли локализация
Есть ли активность
Если диагноз предварительный - назначено ли дообследование
```

### 12.2 Клинические критерии

Проверить:

```text
Есть ли жалобы, соответствующие диагнозу
Есть ли длительность симптомов
Есть ли частота симптомов, если требуется протоколом
Есть ли объективные признаки
Есть ли отрицательные признаки, если они важны для исключения осложнений
```

### 12.3 Обследования

Проверить:

```text
Выполнены ли обязательные обследования
Назначены ли обязательные обследования, если они еще не выполнены
Есть ли дополнительные обследования по показаниям
Есть ли сроки обследований
Есть ли подготовительные обследования перед процедурой
Есть ли контрольные обследования
```

Важно: если обследование уже выполнено, не считать его отсутствующим.

### 12.4 Лечение

Проверить:

```text
Назначено ли лечение, предусмотренное протоколом
Есть ли препарат или группа препаратов
Есть ли доза
Есть ли кратность
Есть ли путь приема
Есть ли длительность
Соответствует ли доза возрасту
Есть ли противопоказания
Есть ли контроль безопасности
Есть ли немедикаментозные рекомендации
```

### 12.5 Контроль и повторная явка

Проверить:

```text
Есть ли повторная явка
Есть ли срок повторной явки
Есть ли контрольные анализы
Есть ли контрольные инструментальные обследования
Есть ли критерии внепланового обращения
```

---

## 13. SafetyChecker

Создать `safety_checker.py`.

Искать красные флаги:

```text
опухолевое образование
нельзя исключить инвазию
подозрение на злокачественное
флеботромбоз
тромбоз глубоких вен
ТГВ
системная красная волчанка
дискоидная красная волчанка с системными проявлениями
высокая температура
выраженная боль
анемия
железодефицит
кровь в стуле
мелена
потеря веса
одышка
боль в груди
неврологический дефицит
```

Для каждого красного флага проверить:

```text
Есть ли дообследование
Есть ли профильная консультация
Есть ли маршрутизация
Есть ли повторная явка
Есть ли срочность
Есть ли контроль
```

Если красный флаг критический и маршрутизация отсутствует:

```text
overall_status = manual_review_required
```

---

## 14. Система оценки

Создать `scoring.py`.

### 14.1 ScoreBreakdown

```python
class ScoreBreakdown(BaseModel):
    structural_score: float | None
    patient_data_score: float | None
    protocol_match_score: float | None
    diagnosis_score: float | None
    required_exams_score: float | None
    treatment_score: float | None
    safety_score: float | None
    follow_up_score: float | None
    overall_score: float | None
```

### 14.2 Веса

```text
structural_score - 15%
patient_data_score - 10%
protocol_match_score - 15%
diagnosis_score - 20%
required_exams_score - 15%
treatment_score - 15%
safety_score - 5%
follow_up_score - 5%
```

Если какой-то блок неприменим, его вес перераспределить между применимыми блоками.

### 14.3 Статусы

```text
90-100 - compliant
75-89 - mostly_compliant
50-74 - partially_compliant
1-49 - non_compliant
нет данных для оценки - insufficient_data
есть критический красный флаг - manual_review_required
```

### 14.4 Правила снижения оценки

```text
Нет даты консультации - минус по structural_score
Нет возраста - минус по patient_data_score и снижение confidence
Нет диагноза - critical issue
Нет рекомендаций - critical issue
Диагноз со знаком вопроса без дообследования - major issue
Есть красный флаг без маршрутизации - manual_review_required
Препарат без дозы - treatment issue
Препарат без кратности - treatment issue
Препарат без длительности - treatment issue, если длительность требуется
Не найден применимый протокол - protocol_match_score = null, overall_status может быть insufficient_data
```

---

## 15. Итоговый отчет

Создать `report_builder.py`.

### 15.1 Markdown-отчет

Структура:

```text
# Оценка консультативного заключения

## 1. Краткое резюме
- Файл
- Дата консультации
- Пациент
- Возраст
- Пол
- Специальность врача
- Диагнозы
- Подобранные протоколы
- Итоговая оценка
- Итоговый статус

## 2. Проверка структуры КЗ
- Что заполнено
- Что отсутствует
- Что заполнено некорректно

## 3. Данные пациента
- Дата рождения
- Возраст на дату консультации
- Пол
- Беременность, если применимо
- Сопутствующие заболевания
- Аллергии
- Текущие препараты

## 4. Проверка диагноза
- Основной диагноз
- Сопутствующие диагнозы
- Подозрительные диагнозы
- МКБ-коды
- Соответствие диагнозов протоколам

## 5. Применимость протоколов
- Почему выбран протокол
- Почему он применим или не применим
- Ограничения по возрасту/полу/беременности

## 6. Проверка обследований
- Выполненные обследования
- Рекомендованные обследования
- Отсутствующие обязательные обследования
- Дополнительные обследования по показаниям

## 7. Проверка лечения
- Назначения
- Дозы
- Кратность
- Длительность
- Соответствие протоколу
- Замечания

## 8. Красные флаги и безопасность
- Найденные красные флаги
- Проверка маршрутизации
- Проверка контроля
- Требуется ли ручная проверка

## 9. Повторная явка и контроль
- Дата или срок повторной явки
- Контрольные обследования
- Контроль безопасности

## 10. Все замечания
- Critical
- Major
- Minor
- Info

## 11. Источники
- Фрагменты КЗ
- Протоколы
- Страницы
- Разделы
- Цитаты
```

### 15.2 JSON-отчет

Формат:

```json
{
  "consultation_id": "",
  "source_file": "",
  "overall_score": 0,
  "overall_status": "",
  "score_breakdown": {},
  "patient_summary": {},
  "doctor_summary": {},
  "diagnoses": [],
  "matched_protocols": [],
  "structural_assessment": {},
  "protocol_assessment": {},
  "issues": [],
  "critical_issues": [],
  "warnings": [],
  "source_refs": []
}
```

---

## 16. Batch-отчет

Для папки КЗ сформировать таблицу:

```text
file
consultation_date
doctor_specialty
patient_age
patient_sex
diagnoses
matched_protocols
overall_score
overall_status
critical_issues_count
major_issues_count
missing_required_sections
red_flags
manual_review_required
```

Сохранить:

```text
data/reports/kz_checks/batch_summary.csv
data/reports/kz_checks/batch_summary.md
```

---

## 17. Тесты

Создать тесты:

```text
tests/test_kz_parser.py
tests/test_patient_resolver.py
tests/test_diagnosis_extractor.py
tests/test_exam_extractor.py
tests/test_medication_extractor.py
tests/test_protocol_matcher.py
tests/test_requirement_checker.py
tests/test_protocol_compliance_checker.py
tests/test_scoring.py
tests/test_safety_checker.py
```

### 17.1 Проверки парсера

Проверить:

```text
Извлекается дата консультации
Извлекается специальность врача
Извлекается ФИО пациента
Извлекается дата рождения
Рассчитывается возраст
Извлекается диагноз
Извлекается МКБ
Извлекаются жалобы
Извлекается объективный статус
Извлекаются рекомендации
```

### 17.2 Проверки диагнозов

```text
K30. Диспепсия -> icd10_code=K30
L93.0. Дискоидная красная волчанка -> icd10_code=L93.0
I80.1. Флебит и тромбофлебит бедренной вены -> icd10_code=I80.1
Дискоидная красная волчанка ? -> certainty=suspected
Нельзя исключить инвазию -> red_flag=possible_malignancy
undefined -> data_quality_issue
```

### 17.3 Проверки лекарств

```text
Ривороксабан 20 мг раз в день постоянно
Гидроксихлорохин 200 мг по 1 таблетке 2 раза/сутки 2 недели
Преднизолон 5 мг по 12 таб в сутки
С 12.08.24 - Преднизолон 5 мг по 11 таб в сутки
Тримедат форте 1 т 2 раза в день 28 дней
```

### 17.4 Проверки протоколов

```text
Не применять детский протокол к взрослому
Не применять взрослый протокол к ребенку
Не применять протокол для беременных без признаков беременности
Если МКБ отсутствует - подбирать по тексту диагноза
Если диагноз suspected - применять только правила для предварительного/подозрительного диагноза
```

### 17.5 Проверки оценки

```text
Нет диагноза -> critical issue
Нет рекомендаций -> critical issue
Нет возраста -> patient_data_score снижается
Красный флаг без маршрутизации -> manual_review_required
Препарат без дозы -> treatment issue
Препарат без длительности -> treatment issue
Неприменимое правило не снижает оценку
```

---

## 18. Критерии приемки

Задача считается выполненной, если:

1. Команда `check-kz` работает для одного PDF.
2. Команда `check-kz-folder` работает для папки PDF.
3. Из КЗ извлекаются дата, врач, специальность, пациент, дата рождения и возраст.
4. Из КЗ извлекаются жалобы, анамнез, объективный статус, обследования, диагнозы, назначения и рекомендации.
5. Возраст рассчитывается на дату консультации.
6. Пол пациента определяется или ставится `unknown`.
7. Диагнозы со знаком вопроса распознаются как `suspected`.
8. `undefined` распознается как дефект качества.
9. Выполненные и рекомендованные обследования различаются.
10. Лекарства, дозы, кратность и длительность извлекаются.
11. Длинные схемы снижения дозы сохраняются как schedule.
12. Подбор протокола учитывает МКБ, текст диагноза, возраст, пол, беременность и специальность.
13. Детские протоколы не применяются к взрослым.
14. Протоколы для беременных не применяются без признаков беременности.
15. Каждое правило проверяется только если оно применимо к пациенту.
16. Формируется JSON-отчет.
17. Формируется Markdown-отчет.
18. Формируется batch summary.
19. Итоговая оценка содержит breakdown по блокам.
20. Critical issues и warnings выводятся отдельно.
21. Красные флаги обнаруживаются.
22. Красный флаг без маршрутизации переводит отчет в `manual_review_required`.
23. Все выводы имеют ссылки на фрагменты КЗ и протоколов.
24. Если данных недостаточно, ставится `insufficient_data`, а не выдумывается оценка.
25. Есть unit-тесты и они проходят.
26. Существующая функциональность проекта не сломана.

---

## 19. Что не делать

Не нужно:

1. Ставить диагноз пациенту.
2. Давать медицинскую рекомендацию пациенту.
3. Исправлять лечение врача.
4. Считать назначение ошибкой без ссылки на протокол.
5. Применять протокол без проверки возраста и применимости.
6. Снижать оценку за правило, которое неприменимо к пациенту.
7. Игнорировать диагнозы со знаком вопроса.
8. Игнорировать красные флаги.
9. Делать вывод без source refs.
10. Удалять существующий код без аудита.
11. Подключать платные API без отдельного согласования.
12. Делать OCR всех документов на первом этапе.
13. Давать пациенту инструкцию по лечению.

---

## 20. Первая команда для Cursor

После добавления этого файла в проект выполнить:

```text
Изучи docs/cursor_task_kz_compliance_checker.md.
Сначала не пиши код.
Проведи аудит текущего проекта и найди, какие модули уже реализованы.
Создай docs/current_project_audit.md.
Создай docs/kz_compliance_implementation_plan.md.
После этого реализуй модуль проверки КЗ по этапам:
1. модели данных;
2. парсер КЗ;
3. проверка обязательных рубрик;
4. подбор протоколов;
5. проверка соответствия протоколам;
6. scoring;
7. отчеты;
8. тесты.
Не ломай существующую функциональность.
```
