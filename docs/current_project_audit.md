# Аудит текущего проекта (Protocol RAG)

> Документ подготовлен по требованию ТЗ `docs/cursor_task_protocols_and_consultations.md`, раздел 2, шаг 1.
> Цель — зафиксировать текущее состояние перед поэтапным расширением, чтобы **не сломать рабочую функциональность**.

Дата: 2026-05-31. Версия сборки на момент аудита: `2026-05-31-r28-consult-json-truncation-fix`.

---

## 0. Краткое резюме

Проект — это **уже работающее RAG-приложение** для клинических протоколов Минздрава РБ с веб-UI (`index.html`), FastAPI-сервером (`rag_server.py`) и пайплайном структурирования корпуса (`corpus_pipeline/`). Анализ консультативных заключений (КЗ) реализован, но **в «облегчённом» виде**: упор на RAG + LLM-синтез JSON-оценки, с тонким слоем детерминированных правил поверх небольшого набора эвристических фактов.

ТЗ предлагает классический структурный конвейер (`src/` + pydantic-модели + отдельные парсеры/движки). **Создавать параллельное `src/`-приложение не нужно и опасно** — это дублировало бы существующую архитектуру. Правильный путь — **встроить недостающие структурные компоненты в текущую архитектуру** (`clinical_knowledge/`, `consult_review_pipeline.py`), сохранив все рабочие части.

---

## 1. Что уже реализовано

### 1.1. Сбор и структурирование протоколов
- **Загрузчик** `download_minzdrav_protocols.py` — однопроходный HTML-скрейпер ссылок (не рекурсивный crawler). Качает 478 PDF по 24 рубрикам в `minzdrav_protocols/<slug>/`. Манифест `minzdrav_protocols/_manifest.jsonl` (url, slug, filename, sha256, bytes, action), ошибки в `_download_errors.json`. Дедуп по sha256 при `--refresh`.
- **Верификатор рубрик** `verify_minzdrav_rubrics.py` — сверяет 24 ожидаемых slug с живым сайтом.
- **Структурный пайплайн** `corpus_pipeline/` (предпочтительный путь):
  - `pdf_extract.py` — постраничный текст (PyMuPDF), confidence, опциональный OCR (`CORPUS_USE_OCR=1`, Tesseract), карта смещений символ→страница.
  - `section_detect.py` — regex-классификация секций (~17 типов: `diagnosis_formula`, `diagnostic_criteria`, `treatment`, `routing`, и т.д.) + нумерованные заголовки.
  - `passport_build.py` — разбиение PDF на несколько «логических» протоколов (478 PDF → 5121 логических документов).
  - `tables_extract.py` — таблицы через pdfplumber, merge multipage.
  - `chunk_build.py` — чанки по секциям/подпунктам с богатой схемой (page_from/to, section_path, icd10_codes, populations, drugs, durations, …) → `output/chunks/chunks.jsonl` (104 687 чанков).
  - `entities_extract.py` — ICD-10 (валидация по справочнику ВОЗ), популяции, care_setting, препараты, длительности.
  - `protocol_cards.py` → `output/registry/protocol_cards.jsonl` (5121 карточка: specialty, population, care_setting, approval, icd10_primary/all, status).
- **Легаси-трек** (для статического хостинга/браузерного поиска): `extract_corpus.py`→`corpus.json`, `build_chunks.py`→`chunks.json`, `build_structured_index.py`→`structured_index.json`, `build_protocol_meta.py`→`protocol_meta.json`, `build_index.py`→`index.csv`/`protocols.json`, `build_semantic_embeddings.py`→`embeddings.json`.

### 1.2. База клинических знаний (`clinical_knowledge/`)
- **Условия (нозологии)**: JSON в `data/gastro_mvp/conditions/` (22, ручные, богатые) и `data/catalog/conditions/` (63, авто, тонкие). Модель строит `condition_builder.py`; merge при совпадении `condition_id`.
- **Правила**: три источника извлечения — `rules_from_path.py` (шаблоны по пути файла), `rules_from_corpus.py` (regex по тексту чанков), `rules_from_enrichment.py` (LLM-кэш, **пока не используется**). Типы: `diagnosis_formula`, `diagnostic_criterion`, `required_exam`, `keyword_presence`, `population_mismatch`. Severity: `critical`/`warning`/`info`.
- **Реестр условий в коде** `condition_registry.py` — `ConditionDef` (78 нозологий: text_markers, icd_prefixes, card_keywords, path_hints) для подсказок/матчинга.
- **Загрузка/merge в рантайме** `loader.py` (`@lru_cache`): объединяет gastro_mvp + catalog для условий и правил, подмешивает enrichment.
- **Детерминированная проверка** `rule_checker.py` — `run_rule_checker()` с runtime-аугментацией path-правил и фильтрацией auto-правил по сопоставленным PDF.
- **Матчинг карточек** `protocol_match.py` — `match_protocol_cards()` по ICD/population/condition hints/keywords.
- **Покрытие** `data/catalog/build_state.json`: 478 PDF, **444 структурировано (92.9%)**, 63 условия, 448 правил, 0 LLM-обогащений, 24 рубрики.

### 1.3. Анализ КЗ (consult-review)
- **API**: `POST /api/consult-review` (синхронно) и `POST /api/consult-review/stream` (SSE с % прогресса).
- **Оркестратор**: `consult_review_pipeline.py` — `iter_consult_review_pipeline()`.
- **Поток**: извлечение текста PDF (pypdf, без OCR) → клинический «фокус» (Gemini digest или эвристика) → инференс ICD + уточнение запроса → merge ICD из текста КЗ → демография (возраст из ДР) → эвристические факты + rule check + матчинг карточек → strict-allowlist протоколов → RAG-ретрив (+ опц. 2-й проход) → сборка контекста → онко-эвристики → **LLM-синтез JSON-оценки** (Gemini, `SYSTEM_CONSULT_REVIEW_JSON`) → отчёт.
- **Итоговый JSON оценки**: `overall_compliance_pct`, `summary_ru`, `criteria[]` (name_ru, score_pct, comment_ru, conclusion_excerpt, protocol_excerpt, protocol_section, protocol_page), `limitations_ru`, `disclaimer_ru`, `protocol_paths_used`. Overall пересчитывается как среднее по критериям (`_stabilize_overall_compliance`).
- **UI** `index.html` — загрузка PDF, прогресс-бар, зоны риска (≥85 / 70–84 / <70), вывод критериев и фрагментов протоколов.

### 1.4. Инфраструктура
- Тесты: ~28 файлов в `tests/` (retrieve, consult_*, clinical_rules, catalog_build, protocol_cards, …).
- CI: `.github/`, `.pre-commit-config.yaml`, `ruff.toml`, `requirements-*.txt`.
- Деплой: `render.yaml`, `runtime.txt`, `docs/deployment-belarus.md`.

---

## 2. Что работает

1. Загрузка одного КЗ в PDF (текстовый слой) и получение оценки соответствия с прогрессом.
2. Структурирование корпуса 478 PDF → 5121 логических документов → 104k чанков с источниковыми ссылками (страница/секция).
3. RAG-ретрив с инвертированным лексическим индексом, ICD-бустами и строгим allowlist протоколов.
4. Извлечение ICD-10 из текста КЗ и из блока диагноза; матчинг протоколов по ICD/специальности/тексту.
5. Детерминированная проверка части правил (формула диагноза, критерии, обязательные обследования — по ключевым словам) поверх 92.9% структурированных PDF.
6. LLM-синтез проверяемой JSON-оценки с цитатами из протоколов; восстановление усечённого JSON и повтор при обрыве.
7. Демография: расчёт «взрослый/ребёнок» из даты рождения для маршрутизации.
8. Устойчивость SSE: фолбэк на синхронный вызов, явные error-события.

---

## 3. Что не работает / отсутствует (gaps относительно ТЗ)

| ТЗ | Текущее состояние |
|----|-------------------|
| **Pydantic-модели КЗ/правил/отчёта** (разделы 7, 9, 13–19) | Только в тексте ТЗ. В рантайме — **plain `dict`**, без валидации. |
| **Структурное извлечение из КЗ** (раздел 10, 25 пунктов) | Эвристика извлекает только: пол, беременность, один `diagnosis_text`, ICD, до 5 жалоб, text_sample[:2000]. **Нет**: ФИО врача/категории/клиники, нескольких диагнозов, ролей/`certainty` диагнозов, анамнеза/объективного статуса как полей, обследований (performed vs recommended), лекарств (доза/кратность/длительность/схема снижения), даты повторной явки. |
| **Парсер лекарств** (раздел 9 `MedicationItem`, `schedule`) | Отсутствует полностью. |
| **Различение performed/recommended/control обследований** (раздел 15) | Только keyword-presence в малом текстовом блобе. |
| **Расчёт возраста на дату консультации** (раздел 9) | Возраст из ДР есть, но привязка к дате консультации и fallback-warning не оформлены. |
| **ProtocolMatcher с applicability** (раздел 12): не применять детские правила к взрослым, протоколы для беременных — только при беременности | Матчинг есть, но **строгие правила применимости по возрасту/полу/беременности не enforced** в проверке. |
| **ComplianceEngine + типизированные оценки** (разделы 13–18): DiagnosisAssessment, ExamAssessment, TreatmentAssessment, SafetyAssessment, SectionQuality | Нет структурированного движка; всё делегировано LLM-синтезу. |
| **Scoring по 6 блокам с весами** (раздел 19) + статусы `insufficient_data` / `manual_review_required` | Есть `config/compliance_weights.yaml`? — **нет** (`config/` отсутствует). Overall = среднее LLM-критериев; нет детерминированной разбивки по 6 весам и статусов insufficient/manual_review. |
| **SafetyChecker / red flags** (раздел 17) + `config/red_flags.yaml` | Есть онко-эвристики, но нет общего safety-движка и конфигурируемых red flags; нет правила «critical red flag без маршрутизации → manual_review_required». |
| **TemplateBlock (автошаблонные КЗ `>>> L30 ...`)** (раздел 11) | Не распознаётся. |
| **Markdown-отчёт** (раздел 20) | Нет; только JSON-оценка для UI. |
| **Longitudinal / эпизоды лечения** (раздел 21) | Отсутствует. |
| **CLI `src.main analyze-consultation ...`** (раздел 5) | Нет; анализ только через HTTP. |
| **`config/*.yaml`** (раздел 6) | Каталог `config/` отсутствует; настройки — через env-переменные. |
| **`data/examples/consultations/`, `tests/fixtures/consultations/`** | Отсутствуют (есть приватные PDF в `clients_consult/`, не обезличенные). |
| **OCR `ocr_required` индекс-поле** | OCR opt-in; флаг только per-page warning, нет индекс-поля. |
| **Дедуп протоколов по URL между рубриками** | Нет (один PDF в двух рубриках скачается дважды). |
| **Покрытие правилами `khirurgiya`** | 24/57 (42.1%) — основная дыра покрытия. |

---

## 4. Какие модули нужно сохранить (НЕ трогать без необходимости)

Это рабочее ядро. Изменения — только аддитивные.

- `rag_server.py` — RAG-ретрив, API, LLM-вызовы. Менять только точечно (добавить новые поля/ветки, не ломать существующие эндпоинты).
- `index.html` — UI. Расширять, не ломая текущую отрисовку.
- `corpus_pipeline/*` — структурирование корпуса. Стабильно.
- `clinical_knowledge/loader.py`, `condition_registry.py`, `protocol_match.py`, `rule_checker.py`, `rules_from_*` — база знаний.
- `download_minzdrav_protocols.py`, `verify_minzdrav_rubrics.py` — сбор протоколов.
- Артефакты: `output/`, `data/catalog/`, `data/gastro_mvp/`, `*.json`/`*.jsonl` индексы.
- Тесты `tests/` — все должны продолжать проходить.

---

## 5. Какие модули нужно переписать / расширить (аккуратно, обратно-совместимо)

- `clinical_knowledge/consult_facts.py` — **расширить** до полноценного структурного парсера КЗ (или вынести богатую часть в новый модуль, оставив `extract_consult_facts_heuristic` как обёртку для обратной совместимости).
- `clinical_knowledge/rule_checker.py` — добавить enforcement применимости (возраст/пол/беременность) и работу с типизированными фактами; сохранить текущую сигнатуру.
- `clinical_knowledge/protocol_match.py` — добавить `applicability` (applicable/possibly_applicable/not_applicable) и mismatch_reasons; не ломать `match_protocol_cards()`.
- `consult_review_pipeline.py` — интегрировать новые структурные факты/оценки как **дополнительные поля** результата; LLM-синтез оставить как есть (или дополнить детерминированной разбивкой).

---

## 6. Какие модули нужно добавить (новое, не конфликтует)

Размещаем в `clinical_knowledge/` (текущий дом доменной логики), а не в параллельном `src/`:

- `consult_schema.py` — pydantic-модели: `ConsultationDocument`, `PatientContext`, `ConsultationSections`, `ConsultationDiagnosis`, `ExamItem`, `MedicationItem`, `MedicationScheduleStep`, `TemplateBlock`, `ExtractionQuality`, и модели отчёта (`ComplianceReport`, `DiagnosisAssessment`, `ExamAssessment`, `TreatmentAssessment`, `SafetyAssessment`, `SectionQualityAssessment`, `ScoreBreakdown`, `ComplianceIssue`, `SourceRef`).
- `consult_parser.py` — секционный парсер КЗ (заголовки из раздела 10 ТЗ) → `ConsultationDocument`.
- `medication_parser.py` — лекарства: препарат/доза/кратность/длительность/`schedule` снижения дозы.
- `date_parser.py`, `age_sex_resolver.py` — даты консультации/ДР, возраст на дату КЗ, возрастные группы.
- `diagnosis_parser.py` — несколько диагнозов, ICD, role, `certainty` (suspected при «?»/«нельзя исключить»).
- `template_parser.py` — автошаблонные блоки `>>> L30 ...`.
- `compliance_engine.py` — детерминированные оценки (диагноз/обследования/лечение/безопасность/качество) поверх правил + фактов.
- `scoring.py` — 6 блоков с весами, статусы (compliant/…/insufficient_data/manual_review_required).
- `safety_checker.py` — red flags из конфига, правило critical→manual_review.
- `consult_report.py` — JSON + Markdown отчёт (раздел 20).
- `config/` — `compliance_weights.yaml`, `red_flags.yaml`, (опц.) `consultation_section_patterns.yaml`, `medication_patterns.yaml`.
- `tests/fixtures/consultations/` + новые тесты (раздел 24) на **обезличенных** примерах.

---

## 7. Какие изменения могут сломать текущую функциональность (риски)

1. **Изменение сигнатур** `extract_consult_facts_heuristic` / `run_rule_checker` / `match_protocol_cards` → сломает `consult_review_pipeline.py` и тесты. → Только аддитивные/опциональные параметры.
2. **Изменение формы результата** consult-review (удаление/переименование полей) → сломает `index.html` и `tests/test_consult_*`. → Только **добавлять** поля (`structured_extraction`, `compliance`, `safety`, `report_markdown`), не трогать существующие.
3. **Жёсткое введение pydantic-валидации** в горячем пути → исключения на «грязных» КЗ остановят анализ. → Валидация с `model_construct`/мягкими дефолтами; ошибка одного блока не валит весь анализ (требование ТЗ 4.6).
4. **Новые тяжёлые зависимости** (pdfplumber, python-docx, regex, dateutil) в рантайме сервера → раздувание/конфликты на Render. → Добавлять только в dev/pipeline requirements, рантайм-парсер КЗ держать на stdlib+pypdf, опциональные импорты с фолбэком.
5. **Принудительный enforcement applicability** может занижать оценки там, где раньше LLM «прощал». → Вводить как отдельный детерминированный слой рядом с LLM, не подменяя текущий `overall_compliance_pct` по умолчанию (фича-флаг).
6. **Скачивание новых протоколов / перезапуск корпуса** → может изменить индексы и сломать кэш. → Не запускать в рамках этой задачи без явного согласования.

---

## 8. План безопасного рефакторинга

**Принципы:**
- Только **аддитивные** изменения форм результата и сигнатур (новые опциональные параметры/поля).
- Каждый новый модуль покрывается тестами до интеграции в горячий путь.
- Новая структурная логика включается **за фича-флагом** (env), по умолчанию поведение не меняется, пока не подтверждена стабильность.
- pydantic-модели — «мягкие» (дефолты, `extra=ignore`), сбой парсинга не валит batch.
- Тяжёлые парсеры (docx/pdfplumber) — опциональные импорты; рантайм КЗ-парсера на stdlib + pypdf.
- После каждого этапа: `ruff` + `pytest`, поднятие `BUILD_VERSION`, commit + `git push`.

**Порядок** (детали — в `docs/implementation_plan.md`):
1. Модели (`consult_schema.py`) — изолированно, без интеграции.
2. Парсеры КЗ (секции, диагнозы, лекарства, даты, шаблоны) — изолированно + юнит-тесты на fixtures.
3. Расширение matcher applicability + safety_checker + config.
4. compliance_engine + scoring (детерминированный слой).
5. report_generator (JSON+MD).
6. Интеграция в `consult_review_pipeline.py` как доп. поля результата (за флагом), затем подключение в UI.
7. Обезличенные fixtures + тесты по разделу 24, проверка критериев приёмки (раздел 25).

**Что НЕ делаем в этой задаче** (ТЗ раздел 26): не создаём параллельное `src/`-приложение, не запускаем массовый OCR, не подключаем платные API, не пересобираем корпус без согласования, не ломаем существующий UI/API.
