# План реализации модуля проверки КЗ (KZ Compliance Checker)

> Основан на `docs/cursor_task_kz_compliance_checker.md` и аудите `docs/current_project_audit.md`.  
> Принцип: **расширять `clinical_knowledge/`**, не дублировать `src/consultation_compliance/`.

---

## Этапы (соответствие запросу)

### Этап 1. Модели данных

**Файл:** `clinical_knowledge/consult_schema.py`

- Добавить поля в `ConsultationSections`: `header`, `consent_text`, `life_history`, `non_drug_recommendations`, `routing`, `doctor_signature`, `consultation_purpose`.
- Расширить `ScoreBreakdown`: `structural_score`, `patient_data_score`, `follow_up_score` (старые поля сохраняются).
- Добавить `StructuralAssessment`, `ProtocolAssessment`, `KzRequirementIssue`.
- Alias `KzComplianceReport = ComplianceReport` + поля `source_file`, `structural_assessment`, `protocol_assessment` в JSON через `report_to_json`.

**Критерий готовности:** `test_consult_schema.py` проходит; новые поля опциональны (`None` по умолчанию).

---

### Этап 2. Парсер КЗ

**Файл:** `clinical_knowledge/consult_parser.py`

- Новые заголовки секций: цель консультации, анамнез жизни, маршрутизация, согласие, подпись врача, немедикаментозные рекомендации.
- Извлечение `doctor_position` / категории при наличии паттерна.
- Флаг `has_unparsed_medication_schedule` при длинных схемах снижения дозы.

**Критерий:** существующие тесты `test_consultation_parser.py` не ломаются.

---

### Этап 3. Проверка обязательных рубрик

**Новый файл:** `clinical_knowledge/requirement_checker.py`

- Загрузка правил из `config/kz_requirements.yaml` (fallback в коде).
- Tier: `required`, `conditional`, `recommended`.
- Required (§9.1): дата, специальность, пациент, возраст/ДР, диагноз, объективный статус, рекомендации, ФИО/подпись врача.
- Conditional (§9.2): локальный статус по специальности, МКБ, follow-up при контроле, маршрутизация при red flags.
- Выход: `StructuralAssessment`, список `ComplianceIssue`, scores `structural_score`, `patient_data_score`.

**Интеграция:** вызов из `compliance_engine.build_compliance_report()` **до** scoring.

**Тесты:** `tests/test_requirement_checker.py`.

---

### Этап 4. Подбор протоколов

**Файлы:** `protocol_match.py`, `applicability.py` — **без breaking changes**.

- Документировать контракт; при необходимости заполнять `matched_condition` из condition registry.
- `consult_analysis.py` — без изменений логики, только прокидывание assessment.

**Тесты:** существующие `test_protocol_matcher.py`.

---

### Этап 5. Проверка соответствия протоколам

**Файлы:** `compliance_engine.py`, `rule_checker.py`

- Подключить issues из `requirement_checker`.
- `_treatment_assessments`: штраф за missing dose/frequency/duration → `treatment_score` (не `None` при наличии meds).
- `_exam_assessments`: учитывать `performed_exams` — не считать missing, если exam выполнен.
- `ProtocolAssessment` — сводка по top match + rules_compliance_pct.

**Тесты:** расширить `test_compliance_engine.py`.

---

### Этап 6. Scoring

**Файлы:** `scoring.py`, `config/compliance_weights.yaml`, `consult_config.py`

- 8 блоков по ТЗ §14.2 (15/10/15/20/15/15/5/5).
- Обратная совместимость: если новые ключи в yaml — использовать их; иначе fallback на 6 блоков.
- `follow_up_score` из наличия follow_up + requirement issues.

**Тесты:** `tests/test_scoring.py`.

---

### Этап 7. Отчёты

**Файлы:** `consult_report.py`, `clinical_knowledge/batch_runner.py`, `scripts/analyze_consultation.py`

- MD: секции «Структура КЗ», «Данные пациента», «Замечания по severity», «Follow-up».
- JSON: `structural_assessment`, `protocol_assessment`, `patient_summary`, `doctor_summary`.
- Batch: `data/reports/kz_checks/batch_summary.csv` + `.md`.
- CLI alias: `--check-kz` / env `KZ_REPORTS_DIR`.

**Критерий:** §16 таблица колонок.

---

### Этап 8. Тесты

| Новый/обновлённый файл | Покрытие |
|------------------------|----------|
| `test_requirement_checker.py` | required/conditional/critical |
| `test_scoring.py` | 8 блоков, insufficient_data |
| `test_safety_checker.py` | manual_review, red flags |
| `test_batch_runner.py` | CSV summary |
| fixtures | gastro, derma suspected, surgery redflag |

**Прогон:** `pytest tests/test_requirement_checker.py tests/test_scoring.py tests/test_compliance_engine.py tests/test_consultation_parser.py`.

---

## Порядок коммитов (рекомендуемый)

1. `docs/` — аудит + план (этот PR).
2. `requirement_checker` + schema + config yaml.
3. `compliance_engine` + scoring 8 blocks.
4. `batch_runner` + report sections + CLI.
5. Tests + BUILD_VERSION bump.

---

## Риски и митигация

| Риск | Митигация |
|------|-----------|
| Сломать consult-review UI | Structured analysis аддитивен; LLM % через `consult_overall_score` |
| Ложные critical на пустых КЗ | `has_content` gate в engine |
| Дублирование с rule_checker | requirement_checker = **структура КЗ**; rule_checker = **протокол** |
| PyYAML отсутствует на prod | Дефолты в `consult_config.py` |

---

## Не входит в этот этап

- OCR всех PDF
- LLM-enrichment правил (`rules_from_enrichment.py` кэш пуст)
- Longitudinal / эпизоды лечения
- Полное сравнение доз с протокольными таблицами (только эвристики missing fields)
