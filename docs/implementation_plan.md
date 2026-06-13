# План внедрения (Protocol RAG → структурный анализ КЗ)

> По ТЗ `docs/cursor_task_protocols_and_consultations.md`, раздел 2, шаг 2.
> Предпосылки и риски - в `docs/current_project_audit.md`.

## Принципы (обязательны на каждом этапе)
- **Не ломать существующее**: формы результата и публичные сигнатуры меняем только аддитивно.
- **Изоляция**: новый код пишется и тестируется отдельно, потом включается в горячий путь за фича-флагом (env).
- **Дом для кода**: `clinical_knowledge/` (не параллельное `src/`). Конфиги - в `config/`.
- **Мягкие pydantic-модели**: дефолты, ошибка парсинга не валит batch (ТЗ 4.6).
- **Рантайм-зависимости** сервера не утяжеляем: docx/pdfplumber - опциональные импорты с фолбэком; КЗ-парсер работает на stdlib + pypdf (уже в зависимостях).
- **Дисциплина завершения этапа**: `ruff check` + `pytest` зелёные → поднять `BUILD_VERSION` → commit → `git push origin`.
- **Каждый вывод проверяем** (ТЗ 27): любой assessment несёт `source_refs` / evidence.

---

## Этап 1 - Модели данных
**Файл:** `clinical_knowledge/consult_schema.py` (новый, pydantic v2).

Модели (по ТЗ 7, 9, 11, 13-19):
- Общие: `SourceRef`, `ComplianceIssue`, `ApplicabilityFilter`.
- КЗ: `PatientContext`, `ConsultationSections`, `ConsultationDiagnosis`, `ExamItem`, `MedicationScheduleStep`, `MedicationItem`, `FollowUpItem`, `TemplateBlock`, `ExtractionQuality`, `ConsultationDocument`.
- Оценка: `DiagnosisAssessment`, `ExamAssessment`, `TreatmentAssessment`, `SafetyAssessment`, `SectionQualityAssessment`, `ScoreBreakdown`, `ProtocolMatchResult`, `ComplianceReport`.

Требования: `model_config = ConfigDict(extra="ignore")`, все поля с дефолтами, `Literal`-перечисления как в ТЗ. Без побочных эффектов и тяжёлых импортов.

**Тест:** `tests/test_consult_schema.py` - конструирование, дефолты, сериализация `model_dump()`.
**Интеграция:** нет (только модели).

---

## Этап 2 - Парсер КЗ
**Файлы:** `consult_parser.py`, `date_parser.py`, `age_sex_resolver.py`, `diagnosis_parser.py`, `template_parser.py`.

- `date_parser.py`: даты ДД.ММ.ГГ(ГГ), словесные; дата/время консультации.
- `age_sex_resolver.py`: пол; возраст **на дату консультации** (ДР+дата КЗ), fallback на текущую дату с warning; возрастные группы newborn/infant/child/adult/elderly.
- `diagnosis_parser.py`: несколько диагнозов; ICD-10 (через существующий `extract_icd10`); `diagnosis_role` (primary при «основной»); `certainty=suspected` при «?», «подозрение», «нельзя исключить», «вероятно».
- `template_parser.py`: блоки `>>> <ICD> <текст>:` + `* ОБСЛЕДОВАНИЯ ОБЯЗАТЕЛЬНЫЕ/ДОПОЛНИТЕЛЬНЫЕ:` → `TemplateBlock`.
- `consult_parser.py`: разбор по заголовкам (Жалобы/Анамнез/Аллергоанамнез/Объективный статус/Локальный статус/Данные обследований/Диагноз/Рекомендации по обследованию/лечению/Дата повторной явки) → `ConsultationSections` + сбор всего в `ConsultationDocument` (+ `ExtractionQuality`).

**Тесты:** `tests/test_consultation_parser.py`, `tests/test_diagnosis_parser.py` (кейсы ТЗ 24).
**Интеграция:** нет (изолированно).

---

## Этап 2b - Лекарства, обследования, качество
**Файлы:** `medication_parser.py`, расширение `consult_parser.py`.

- `medication_parser.py`: препарат, доза (value+unit), кратность, длительность, путь; длинные схемы снижения дозы → `schedule[]` (кейсы ТЗ 9: Тримедат, Эспумизан, УДХК, Ривароксабан, Гидроксихлорохин, Преднизолон-снижение). Сохранять `raw_text`, не терять единицы (ТЗ 23).
- Обследования: разделить performed / recommended / control / planned (`ExamItem.status`).
- `ExtractionQuality`: warnings/errors, флаги `has_undefined`, `has_question_mark_diagnosis`, `has_unparsed_medication_schedule`, `has_missing_*`.

**Тест:** `tests/test_medication_parser.py` (кейсы ТЗ 24).
**Интеграция:** нет.

---

## Этап 3 - Парсер протоколов
**Статус:** в основном реализован (`corpus_pipeline/`, `protocol_cards`). В этой задаче - **без пересборки корпуса**.

Только мелкие аддитивные улучшения при необходимости (например, фиксация `ocr_required` как производного поля в карточке - опционально, за флагом). Главный фокус - закрыть дыру `khirurgiya` (42%) дополнительными path-шаблонами в `rules_from_path.py` (аддитивно), без влияния на остальные рубрики.

**Тест:** обновить `tests/test_catalog_build.py` при изменении покрытия.

---

## Этап 4 - Подбор протоколов (applicability)
**Файл:** расширить `clinical_knowledge/protocol_match.py` (+ helper `applicability.py` при необходимости).

Добавить к `match_protocol_cards()` (аддитивно, новые поля результата):
- `applicability`: applicable / possibly_applicable / not_applicable / unknown.
- `match_reasons[]`, `mismatch_reasons[]`.
- Правила ТЗ 12: детский протокол не применяется к взрослому автоматически; протокол «только для беременных» - лишь при беременности; диагноз с «?» → как suspected; несколько диагнозов - отдельно; red flag повышает приоритет.

**Тест:** `tests/test_protocol_matcher.py` (кейсы ТЗ 24: L93.0→дерма, K30→гастро, детский/беременность-фильтры).
**Интеграция:** matcher вызывается, но enforcement в скоринге - этап 5 (за флагом).

---

## Этап 5 - Compliance engine + Safety + Scoring
**Файлы:** `compliance_engine.py`, `safety_checker.py`, `scoring.py`, `config/compliance_weights.yaml`, `config/red_flags.yaml`.

- `safety_checker.py`: red flags из `config/red_flags.yaml` (possible_malignancy/thrombosis/systemic_autoimmune и др.); проверка наличия дообследования/маршрутизации/повторной консультации; `SafetyAssessment`. Правило: критический red flag без маршрутизации ⇒ статус не выше `manual_review_required`.
- `compliance_engine.py`: на входе `ConsultationDocument` + matched protocols + правила; строит `DiagnosisAssessment` / `ExamAssessment` / `TreatmentAssessment` / `SectionQualityAssessment`; **не применяет неприменимые по возрасту/полу правила** (ТЗ 26.6).
- `scoring.py`: 6 блоков с весами (protocol_match 15, diagnosis 20, exams 20, treatment 20, safety 15, documentation 10); статусы по порогам; `insufficient_data` при нехватке данных (не занижать искусственно); `manual_review_required` при критическом red flag.

**Тесты:** `tests/test_compliance_engine.py` (кейсы ТЗ 24: обязательные обследования, performed≠missing, «?»→suspected, red flag→manual_review, нет возраста→confidence↓ но не всегда compliance↓, неприменимые правила не считаются).
**Интеграция:** доступно как отдельный детерминированный результат; в горячий путь - этап 7 (флаг).

---

## Этап 6 - Scoring
Покрыто этапом 5 (`scoring.py`). Отдельно: верификация порогов/весов из конфигов, граничные случаи (0 правил, нет matched-протоколов).

---

## Этап 7 - Отчёты
**Файл:** `consult_report.py`.

- JSON-отчёт (ТЗ 20) из `ComplianceReport`.
- Markdown-отчёт (8 разделов ТЗ 20): резюме, применимость протокола, диагноз, обследования, лечение, red flags, качество КЗ, источники.
- Каждый блок - с `source_refs`.

**Тест:** `tests/test_consult_report.py` - генерация MD/JSON из синтетического `ComplianceReport`, наличие источников.

---

## Этап 8 - Тесты и fixtures
**Каталог:** `tests/fixtures/consultations/` - **обезличенные** TXT/JSON-примеры (НЕ копировать реальные PDF из `clients_consult/`).

Покрыть: `test_consultation_parser`, `test_diagnosis_parser`, `test_medication_parser`, `test_protocol_matcher`, `test_compliance_engine` (ТЗ 24). Прогон всего `pytest` зелёный.

---

## Этап 9 - Интеграция с существующим UI/API
**Файлы:** `consult_review_pipeline.py`, `rag_server.py`, `index.html`.

- В результат consult-review **добавить новые поля** (не трогая существующие): `structured_extraction` (из `ConsultationDocument`), `compliance` (детерминированный `ComplianceReport`), `report_markdown`. За флагом `CONSULT_STRUCTURED_ANALYSIS=1` (по умолчанию можно включить только сбор, не подмену overall).
- LLM-синтез остаётся; детерминированный слой идёт рядом (для проверяемости и source_refs).
- UI: отдельный блок «Структурный разбор» (диагнозы/ICD/обследования/лекарства/red flags) - аддитивно к текущей отрисовке.
- Опционально: CLI-обёртка `scripts/analyze_consultation.py` (`--file`, `--folder`) для batch (ТЗ 5), переиспользуя пайплайн.
- README: раздел про анализ КЗ.

**Проверка критериев приёмки** (ТЗ 25, пункты 1-30) + «существующая функциональность не сломана» (тесты зелёные, UI работает).

---

## Зависимости (минимизация рисников)
- Рантайм (`requirements-rag.txt`): без новых тяжёлых пакетов; парсеры на stdlib + `re` + `pypdf`.
- Dev/pipeline (`requirements-dev.txt` / `requirements-corpus-pipeline.txt`): при необходимости `python-dateutil`, `pyyaml` (для config). `pyyaml` лёгкий - допустимо и в рантайме для чтения config с фолбэком на встроенные дефолты.
- docx-парсинг (`python-docx`) - опционально, только если появятся .docx КЗ; импорт ленивый.

## Порядок коммитов
Один коммит на этап (или логическую часть), после зелёных тестов, с поднятием `BUILD_VERSION` и `git push origin main`.
