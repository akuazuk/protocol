# Автономное ТЗ для Cursor: качество оценки КЗ, корпус протоколов и единый scorer v3

**Дата:** 2026-07-27  
**Статус:** active  
**Режим исполнения:** автономная ночная сессия до завершения всех автоматизируемых пунктов  
**Финал:** тесты, отчёт, коммит, push отдельной ветки  
**Связанные планы:**  
`2026-07-22-kz-scoring-methodology-v1.md`,  
`2026-07-22-kz-deep-eval-db-task-v1.md`,  
`2026-07-20-protocol-reextract-quality-v1.md`.

---

## 0. Главная команда Cursor

Выполняй задачу автономно и последовательно. Не останавливайся после анализа, плана,
частичного рефакторинга или первого зелёного теста. Продолжай, пока:

1. не реализована вся безопасно автоматизируемая часть этого ТЗ;
2. не выполнены релевантные тесты;
3. не создан итоговый отчёт с метриками «до/после»;
4. изменения не закоммичены;
5. ветка не отправлена в origin.

Не задавай уточняющих вопросов, если можно принять консервативное решение без изменения
медицинского смысла. Если часть задачи требует внешнего секрета, БД, ручного решения
методиста или недоступного API:

- не подделывай результат;
- зафиксируй блокер в итоговом отчёте;
- добавь воспроизводимую команду продолжения;
- переходи к следующему независимому пункту.

Слова «выполнить всё» не разрешают:

- менять медицинские цитаты вручную без источника;
- объявлять auto/LLM-карточки утверждёнными методистом;
- отправлять персональные данные во внешние сервисы;
- отключать safety gates ради роста среднего score;
- удалять пользовательские файлы или незакоммиченные изменения;
- делать force-push;
- коммитить секреты, PDF с ПДн, runtime-кэши и массовые generated artifacts.

---

## 1. Безопасный старт в текущем репозитории

### 1.1. Рабочее дерево уже грязное

На момент постановки задачи в основном worktree есть множество пользовательских
изменений и generated artifacts. Они не принадлежат этой задаче.

Перед работой:

```bash
git status --short > /tmp/protocol-kz-eval-preexisting-status.txt
git diff --stat
git log -5 --oneline
```

Не использовать:

```bash
git reset --hard
git checkout -- .
git clean -fd
git stash
```

### 1.2. Работать в отдельном clean worktree

Чтобы не смешать ночную задачу с текущими изменениями:

```bash
TASK_WORKTREE=/private/tmp/protocol-kz-evaluation-v3
git worktree add -b codex/kz-evaluation-quality-v3 "$TASK_WORKTREE" HEAD
cp docs/plans/2026-07-27-kz-evaluation-quality-overnight-v1.md \
  "$TASK_WORKTREE/docs/plans/2026-07-27-kz-evaluation-quality-overnight-v1.md"
cd "$TASK_WORKTREE"
```

Если ветка уже существует, не перезаписывать её. Проверить состояние и продолжить
существующую ветку либо создать ветку с суффиксом `-v2`.

Все изменения, тесты, коммиты и push выполнять из clean worktree.

### 1.3. Зафиксировать baseline

Сохранить в итоговый отчёт:

- commit SHA;
- Python version;
- текущий `BUILD_VERSION`;
- версии scorer/config;
- результаты целевых тестов до изменений;
- существующие corpus-quality метрики.

Не копировать в clean worktree незакоммиченные данные из основного worktree, кроме
этого ТЗ. Если для анализа нужен generated report из основного worktree, читать его
как внешний вход, но не коммитить автоматически.

---

## 2. Контекст и доказанные проблемы

### 2.1. Несколько конкурирующих итогов

В системе одновременно существуют:

- `structured_analysis.compliance.overall_score`;
- `alignment_mean_score`;
- `review.overall_compliance_pct`;
- deep `overall_pct`;
- `send_gate.gate_score`.

UI и API могут показывать разные проценты для одного КЗ. Gate берёт минимум headline
и structural, но это не представлено единым версионированным контрактом.

### 2.2. Недостатки текущего structural scorer

1. Рекомендуемые и условные заполненные поля могут компенсировать отсутствие
   обязательных полей в `structural_score`.
2. Блоки со значением `None` исключаются из знаменателя, из-за чего неполное КЗ иногда
   получает завышенный балл.
3. `protocol_match_score` дискретный и грубый: примерно 90/65/20.
4. Лечение часто оценивается как полнота записи дозы/кратности/длительности, а не как
   соответствие конкретному режиму протокола.
5. Alignment строже baseline, но не подключён к overall по умолчанию.
6. Deep scorer A/B/C/D работает параллельно live scorer и не является каноническим.
7. Confidence, evidence coverage и clinical score смешаны концептуально.

### 2.3. Проблемы корпуса протоколов

Текущие метрики:

- PDF: 478;
- protocol summary: около 475–477;
- valid summary, включая warnings: 240;
- approved методистом: 0;
- needs review: 462;
- chunks: 59 045;
- chunks с замечаниями: 26 772;
- weak section title: 11 811;
- too long: 6 427;
- truncated list: 4 908;
- clinical content classified as body: 4 043;
- empty entities: 2 423;
- preamble leak: 642.

Покрытие summary excerpts:

- diagnosis: 295/477;
- exams: 181/477;
- treatment: 415/477;
- follow-up: 223/477.

`rules_coverage_report` показывает правила почти для каждого PDF, но значительная
часть создана из rich tables, path templates и эвристик. Наличие одного правила не
равно пригодности протокола для штрафующей оценки.

---

## 3. Цель ночной итерации

За одну автономную итерацию создать безопасный фундамент scorer v3:

1. единый контракт результата;
2. явные `score`, `coverage`, `confidence`, `risk` и `provenance`;
3. trust levels для требований протокола;
4. запрет жёстких штрафов по недоверенным правилам;
5. исправление structural score;
6. versioned shadow scorer без переключения production gate;
7. аудит корпуса, пригодный для приоритизации ручного ревью;
8. regression tests;
9. документацию миграции;
10. коммит и push.

Не требуется за одну ночь вручную утвердить 477 протоколов или создать настоящий
methodist gold set. Требуется построить код и очереди, которые делают эту работу
управляемой и измеримой.

---

## 4. Целевая архитектура

```text
КЗ / FHIR / PDF
  -> извлечение текста
  -> классификация типа документа
  -> ConsultationDocument + provenance spans
  -> protocol applicability
  -> trusted atomic requirements
  -> evidence matcher
  -> axes A/B/C/D
  -> coverage + confidence
  -> risk gate
  -> calibrated canonical result
  -> UI / API / send gate

PDF протокола
  -> layout / section / table extraction
  -> ProtocolKnowledgeModel
  -> automatic validation
  -> trust level
  -> methodist review
  -> versioned atomic requirements
```

Главный инвариант:

> Scorer не имеет права жёстко штрафовать КЗ по правилу, если применимость правила,
> источник и медицинский смысл не подтверждены достаточным trust level.

---

## 5. Workstream A — единый контракт `KzEvaluationResultV3`

### 5.1. Добавить модели

Создать отдельный модуль, предпочтительно:

```text
clinical_knowledge/kz_evaluation_v3.py
```

Либо разделить:

```text
clinical_knowledge/kz_evaluation_schema.py
clinical_knowledge/kz_evaluation_engine.py
```

Минимальная схема:

```json
{
  "schema_version": "3.0",
  "scorer_version": "2026-07-27.1",
  "score_pct": 64.0,
  "status": "review",
  "axes": {
    "documentation": 78.0,
    "clinical_concordance": 51.0,
    "safety": 72.0,
    "regulatory": 80.0
  },
  "coverage": {
    "overall": 0.74,
    "documentation": 0.95,
    "clinical_concordance": 0.61,
    "safety": 0.80,
    "regulatory": 0.70
  },
  "confidence": {
    "overall": 0.69,
    "document_parse": 0.92,
    "protocol_match": 0.78,
    "evidence_match": 0.55,
    "protocol_knowledge": 0.60
  },
  "risk": {
    "worst_severity": "P1",
    "cap_applied": true,
    "cap_value": 60.0,
    "reasons": []
  },
  "protocols": [],
  "findings": [],
  "provenance": {
    "corpus_version": "...",
    "rules_version": "...",
    "weights_version": "...",
    "build_version": "..."
  },
  "legacy": {}
}
```

### 5.2. Требования к контракту

- Все поля имеют безопасные defaults.
- Никаких `NaN`/`Infinity`.
- Значения score: 0–100.
- Coverage/confidence: 0–1.
- Отсутствие данных выражается `None`, а не нулём.
- `legacy` содержит старые scores только для shadow-сравнения.
- Контракт сериализуется Pydantic и стабилен в API.
- Добавить unit tests на сериализацию и границы.

### 5.3. Не ломать API

Добавить v3 аддитивно:

```json
{
  "evaluation_v3": {...},
  "structured_analysis": {...},
  "review": {...},
  "send_gate": {...}
}
```

Не удалять старые поля в этой итерации.

---

## 6. Workstream B — trust levels для правил протокола

### 6.1. Ввести уровни доверия

```text
A = approved_by_methodist
B = validated_with_source
C = auto_extracted
D = heuristic
```

Для каждого правила хранить:

```json
{
  "trust_level": "B",
  "review_status": "reviewed",
  "extraction_method": "llm_extracted",
  "source_quote_verified": true,
  "applicability_verified": true,
  "penalty_allowed": true
}
```

### 6.2. Консервативное сопоставление текущих источников

Пример начального mapping:

- `review_status=approved` -> A;
- валидная summary, подтверждённая цитата и приемлемый review status -> B;
- `llm_extracted`, `auto_extracted`, summary без review -> C;
- `path_template`, `rich_table`, inferred path condition, fallback -> D.

Не повышать автоматически C/D до B.

### 6.3. Политика влияния на score

- A/B: могут создавать `missing`, штраф и risk finding.
- C: могут создавать `needs_human`, подсказку и снижать confidence.
- D: только retrieval/routing hint; не штрафуют.
- Critical/P0 от C/D не блокирует send gate без независимого подтверждения.
- Наличие точной цитаты обязательно для A/B penalty.

### 6.4. Интеграция

Проверить и при необходимости изменить:

- `clinical_knowledge/rule_model.py`;
- `clinical_knowledge/rule_checker.py`;
- `clinical_knowledge/rules_from_summary.py` или соответствующий summary adapter;
- `clinical_knowledge/rules_from_corpus.py`;
- `clinical_knowledge/rules_from_path.py`;
- `clinical_knowledge/evidence_map.py`;
- `clinical_knowledge/protocol_compliance_checker.py`.

Добавить diagnostics:

```json
{
  "rules_total": 20,
  "rules_penalty_eligible": 6,
  "rules_advisory": 8,
  "rules_heuristic": 6
}
```

---

## 7. Workstream C — исправление structural score

### 7.1. Обязательные поля не компенсируются optional-полями

В `requirement_checker.py` считать отдельно:

```text
required_score
conditional_score
recommended_score
patient_data_score
```

Рекомендуемые поля не входят в numerator обязательного score.

Предлагаемая базовая формула:

```text
documentation_score =
  0.70 * required_completion
  + 0.20 * applicable_conditional_completion
  + 0.10 * recommended_completion
```

Если отсутствует критическое обязательное поле, применить явный cap, а не косвенный
штраф, например:

- нет диагноза -> documentation cap 45;
- нет рекомендаций -> cap 55;
- нет объективного статуса на первичном приёме -> cap 65;
- пустое/нечитаемое КЗ -> insufficient data.

Точные cap оформить конфигом и тестами. Не менять safety cap.

### 7.2. Coverage-aware scoring

`None`-блоки больше не должны бесследно исчезать.

Вычислять:

```text
score = weighted mean доступных проверок
coverage = сумма весов проверенных требований / сумма потенциально применимых весов
```

Статус:

- coverage >= 0.80: обычный;
- 0.50–0.79: score с `limited_evidence`;
- coverage < 0.50: `insufficient_evidence`;
- critical safety finding действует независимо от coverage.

Не переключать legacy status на новую политику без shadow-сравнения.

### 7.3. Разделить score и confidence

Confidence не должен повышать или снижать клинический score скрыто.

Он влияет на:

- статус надёжности;
- право hard gate;
- необходимость ручного ревью;
- UI-дисклеймер.

---

## 8. Workstream D — единый scorer поверх A/B/C/D

### 8.1. Использовать существующий deep engine как основу, не дублировать

Расширить `kz_deep_eval.py` или вынести общие функции в v3 engine.

Оси:

### A. Documentation

- required sections;
- patient data;
- первичный/повторный приём;
- реквизиты;
- качество извлечения;
- непротиворечивость разделов.

### B. Clinical concordance

- клинические данные -> диагноз;
- диагноз -> МКБ;
- диагноз -> применимый протокол;
- required exams;
- conditional exams;
- treatment;
- monitoring;
- follow-up.

### C. Safety

- red flags;
- маршрутизация;
- подозрение на ЗНО;
- лекарственные взаимодействия;
- high-alert drugs;
- double NSAID;
- противопоказания;
- мониторинг;
- опасная доза;
- беременность/возраст.

### D. Regulatory

- постановление №55;
- ЦИСЗ/FHIR readiness;
- обязательные реквизиты;
- подпись/идентификация;
- consent, если применимо.

### 8.2. Risk gate

- P0 -> hard cap и `critical`;
- P1 -> cap/review;
- слабая ось не маскируется сильным средним;
- P0/P1 должны иметь evidence и trust level;
- finding без достаточного evidence -> `needs_human`, не факт нарушения.

### 8.3. Shadow mode

Добавить env:

```text
KZ_EVALUATION_V3_ENABLED=1
KZ_EVALUATION_V3_PRIMARY=0
KZ_EVALUATION_V3_GATE=0
```

В этой итерации defaults:

```text
enabled=true
primary=false
gate=false
```

Production score и gate не переключать автоматически.

---

## 9. Workstream E — улучшение protocol applicability

### 9.1. Разделить уверенности

Для каждого protocol match:

```json
{
  "applicability_confidence": 0.84,
  "retrieval_confidence": 0.67,
  "population_match": true,
  "care_setting_match": true,
  "specialty_match": true,
  "version_current": true,
  "penalty_eligible": true
}
```

### 9.2. Жёсткие инварианты

- детский КП не штрафует взрослое КЗ;
- стационарный КП не штрафует амбулаторное КЗ;
- реабилитационный КП не заменяет диагностический;
- общий организационный КП не считается disease-specific;
- protocol fallback не штрафует;
- несколько диагнозов связываются с отдельными protocol contexts;
- при конфликте версий используется актуальная версия либо manual review;
- при `applicability_confidence < 0.75` protocol requirements advisory-only.

### 9.3. Тестовые кейсы

Добавить regression tests:

- взрослый vs детский КП;
- ЛОР J06 vs чужой пульмонологический/стационарный КП;
- неврология vs позвоночник;
- реабилитация vs диагностика;
- беременность;
- suspected oncology;
- несколько диагнозов.

---

## 10. Workstream F — аудит и приоритизация корпуса

### 10.1. Создать воспроизводимый аудит

Добавить команду:

```bash
python -m scripts.audit_kz_protocol_knowledge \
  --json data/ml/reports/kz_protocol_knowledge_audit_latest.json \
  --markdown data/ml/reports/kz_protocol_knowledge_audit_latest.md
```

Отчёт должен содержать:

- total protocols;
- active/current/obsolete/unknown version;
- extraction status;
- review status;
- trust levels;
- quote verification;
- applicability coverage;
- diagnosis criteria coverage;
- required exams coverage;
- conditional exams coverage;
- treatment coverage;
- dose/route/frequency/duration coverage;
- red flags coverage;
- monitoring coverage;
- follow-up coverage;
- broken/truncated/table issues;
- penalty-eligible rules;
- advisory rules;
- protocols without any safe penalty rule.

### 10.2. Не считать «есть хотя бы одно правило» качественным покрытием

Ввести метрики:

```text
protocol_structured_coverage_pct
penalty_eligible_coverage_pct
source_verified_coverage_pct
methodist_approved_coverage_pct
```

### 10.3. Очередь методиста

Сформировать приоритет:

```text
priority =
  real_mis_frequency
  * clinical_risk
  * missing_structure_factor
  * protocol_match_error_factor
```

Если MIS frequency недоступна в clean worktree, использовать доступные агрегаты;
не переносить ПДн. При отсутствии данных добавить параметр `--mis-summary`.

Выход:

```text
data/ml/reports/kz_protocol_methodist_queue_latest.json
data/ml/reports/kz_protocol_methodist_queue_latest.md
```

Не коммитить огромные файлы. Коммитить только компактный агрегат и схему/скрипт.

---

## 11. Workstream G — каноническая knowledge model протокола

### 11.1. Добавить модель без массовой миграции всех PDF

Создать Pydantic/dataclass schema:

```text
ProtocolKnowledgeDocument
ConditionDefinition
AtomicRequirement
Applicability
MedicationRegimen
SourceEvidence
KnowledgeReview
```

`AtomicRequirement`:

- `requirement_id`;
- `type`;
- `obligation`;
- `canonical_item_id`;
- `text_ru`;
- `applicability`;
- `source`;
- `trust`;
- `review_status`;
- `extraction_confidence`;
- `penalty_allowed`.

### 11.2. Адаптер текущих summary

Сделать converter:

```text
ProtocolSummary -> ProtocolKnowledgeDocument
```

Он обязан:

- сохранять source refs;
- не придумывать обязательность;
- маркировать auto fields уровнем C;
- маркировать path/rich-table heuristics уровнем D;
- выдавать validation diagnostics;
- сохранять condition scope.

### 11.3. Validator

Knowledge document не penalty-ready, если:

- нет source quote;
- quote не подтверждена;
- нет applicability;
- правило оборвано;
- таблица потеряла контекст;
- condition неизвестен;
- review/trust ниже B.

Добавить CLI validator и tests.

---

## 12. Workstream H — parser КЗ и provenance

В пределах ночной итерации сделать безопасный фундамент:

1. добавить span/source metadata хотя бы для diagnosis, exams, treatment, follow-up;
2. не ломать существующий `ConsultationDocument`;
3. добавить `document_type` и confidence;
4. различать `performed`, `recommended`, `planned`, `negative`;
5. вынести provenance в аддитивное поле либо отдельную map;
6. добавить tests на отрицания и предварительный диагноз.

Не переписывать весь parser целиком, если это ставит под угрозу стабильность.

Обязательные regression cases:

- «МРТ не выполнялось, рекомендовано»;
- «антибиотики не показаны»;
- «подозрение на...»;
- «диагноз исключён»;
- несколько документов в хронологии;
- плохой OCR/короткий текст.

---

## 13. Workstream I — лекарственная терапия

### 13.1. Разделить типы находок

```text
documentation_gap
protocol_mismatch
safety_warning
insufficient_context
needs_human
```

### 13.2. Нормализация

Использовать существующие:

- `drug_normalizer.py`;
- `medication_parser.py`;
- `medication_safety.py`;
- `data/drug_safety/`.

Не объявлять dose mismatch, если:

- не определено действующее вещество;
- нет trustworthy regimen;
- неизвестны масса/СКФ/возраст при зависимой дозе;
- источник C/D;
- confidence ниже порога.

### 13.3. Ночной deliverable

- единая структура medication finding;
- trust-aware penalty;
- tests на missing dose vs dangerous dose;
- double NSAID остаётся safety finding;
- существующие safety tests не регрессируют.

Полное наполнение всех дозировок протоколов относится к методистскому продолжению.

---

## 14. Workstream J — API, UI и gate

### 14.1. API

Добавить `evaluation_v3` в:

- L0;
- L1;
- L2;
- JSON/FHIR endpoints;
- при возможности patient API только после sanitize и без B2B gate fields.

### 14.2. UI

В этой итерации не делать большой редизайн. Добавить shadow/debug представление только
для методиста или feature flag:

- canonical v3 score;
- legacy score;
- coverage;
- confidence;
- trust diagnostics;
- risk cap;
- scorer/corpus version.

Не показывать два конкурирующих headline обычному врачу.

### 14.3. Gate

`KZ_EVALUATION_V3_GATE=0` по умолчанию.

Подготовить функцию gate v3, но не включать её:

- hard block только для подтверждённого P0/A-B evidence;
- low score без critical -> override/review;
- low confidence -> review required;
- low coverage -> review required;
- C/D findings не блокируют.

---

## 15. Workstream K — gold set и калибровка

Нельзя создать экспертный gold без методистов. За ночь подготовить инфраструктуру:

1. schema gold annotation;
2. deterministic sample builder;
3. double-annotation fields;
4. adjudication status;
5. validator;
6. evaluator QWK/MAE/harm recall/false critical rate;
7. пример полностью синтетических записей;
8. инструкцию методисту.

Целевая будущая выборка:

- 800–1200 КЗ;
- специальности;
- типы приёма;
- score bands;
- red flags;
- high-risk medications;
- хорошие и плохие документы.

Метрики:

- inter-rater weighted kappa >= 0.65;
- QWK scorer vs consensus >= 0.70;
- MAE <= 10;
- P0 recall >= 0.95;
- false critical <= 5%;
- top-1 protocol accuracy >= 90%;
- source refs for penalties = 100%.

LLM labels хранить как proxy, не gold.

---

## 16. Тестовая стратегия

### 16.1. Обязательные новые тесты

Создать отдельные файлы либо расширить существующие:

```text
tests/test_kz_evaluation_v3.py
tests/test_rule_trust.py
tests/test_kz_coverage_scoring.py
tests/test_protocol_knowledge_model.py
tests/test_protocol_knowledge_audit.py
tests/test_kz_v3_gate.py
```

Проверить:

1. optional не компенсирует required;
2. `None` увеличивает uncertainty/снижает coverage, а не даёт скрытое преимущество;
3. C/D rule не штрафует;
4. A/B rule со source ref штрафует;
5. C/D critical не блокирует gate;
6. A/B P0 применяет cap;
7. low coverage -> review/insufficient evidence;
8. low confidence -> review;
9. adult/child mismatch;
10. inpatient/outpatient mismatch;
11. fallback protocol advisory-only;
12. source quote required;
13. JSON schema stable;
14. legacy API не сломан.

### 16.2. Запуск тестов

Сначала узкие:

```bash
.venv/bin/pytest -q \
  tests/test_scoring.py \
  tests/test_compliance_gate.py \
  tests/test_consult_alignment.py \
  tests/test_kz_deep_eval.py \
  tests/test_consult_tiering.py \
  tests/test_kz_evaluation_v3.py \
  tests/test_rule_trust.py \
  tests/test_kz_coverage_scoring.py \
  tests/test_protocol_knowledge_model.py \
  tests/test_protocol_knowledge_audit.py \
  tests/test_kz_v3_gate.py
```

Затем:

```bash
.venv/bin/ruff check clinical_knowledge scripts tests rag_server.py consult_review_pipeline.py
.venv/bin/pytest -q
```

Если clean worktree не содержит `.venv`, использовать Python 3.11/3.12 и установить
только проектные requirements. Не менять lock/dependencies без необходимости.

### 16.3. Нельзя «чинить» тесты ослаблением безопасности

Запрещено:

- удалять assertions;
- менять critical на warning ради зелёного теста;
- отключать тесты;
- добавлять blanket `except Exception` в core scorer;
- маскировать ошибку пустым результатом.

---

## 17. Shadow benchmark

Добавить скрипт:

```bash
python -m scripts.compare_kz_evaluation_v3 \
  --fixtures tests/fixtures \
  --json data/ml/reports/kz_evaluation_v3_shadow_latest.json \
  --markdown data/ml/reports/kz_evaluation_v3_shadow_latest.md
```

Минимальные показатели:

- N;
- legacy score mean/median;
- v3 score mean/median;
- delta distribution;
- status changes;
- coverage distribution;
- confidence distribution;
- caps;
- cases where legacy high but v3 low;
- cases where legacy low but v3 insufficient evidence;
- C/D findings excluded from penalties;
- protocol mismatch cases.

Если доступны безопасные агрегированные MIS cases без ПДн — поддержать `--cases`.
Не коммитить raw cases.

---

## 18. Критерии приёмки ночной итерации

### P0 — обязательно до push

- [x] Работа идёт в отдельном clean worktree/ветке (`codex/kz-evaluation-quality-v3`).
- [x] Добавлен `KzEvaluationResultV3` (`clinical_knowledge/kz_evaluation_schema.py`).
- [x] Добавлены score/coverage/confidence/risk/provenance.
- [x] Добавлены trust levels A–D (`clinical_knowledge/rule_trust.py`).
- [x] C/D rules не создают штраф или hard gate (тесты `test_rule_trust`, `test_kz_v3_gate`).
- [x] Optional fields не компенсируют required (`test_kz_coverage_scoring`).
- [x] Добавлен coverage-aware calculation (`score_documentation`/`score_concordance`).
- [x] V3 работает в shadow mode (env-флаги, defaults enabled/primary=0/gate=0).
- [x] Legacy API сохранён (v3 аддитивно, legacy-поля не тронуты).
- [x] Добавлен corpus knowledge audit (`scripts/audit_kz_protocol_knowledge.py`).
- [x] Добавлен Protocol Knowledge schema + summary adapter (`protocol_knowledge_model.py`).
- [x] Добавлены tests для trust/coverage/gate (+ knowledge/audit).
- [x] Узкие тесты зелёные.
- [x] Полный pytest — см. итог сессии/отчёт (baseline failures задокументированы, если есть).
- [x] Ruff зелёный для изменённых файлов.
- [x] Создан итоговый markdown-отчёт (`docs/reports/kz-evaluation-v3-overnight-result-2026-07-28.md`).
- [x] Создан коммит.
- [x] Ветка отправлена в origin.

### P1 — выполнить, если P0 завершён

- [ ] Provenance spans для ключевых полей КЗ (отложено: требует правки parser; §12).
- [x] Trust-aware medication findings (`clinical_knowledge/medication_findings.py`).
- [x] Методистская очередь по качеству протоколов (audit `--queue-*`).
- [x] Shadow comparison script (`scripts/compare_kz_evaluation_v3.py`).
- [x] Feature-flagged методистский UI для v3 diagnostics (data-поле `evaluation_v3` в деталях).
- [x] Gold annotation schema/evaluator (`clinical_knowledge/kz_gold_annotation.py`).

### P2 — не блокирует ночной push

- [ ] Полная ручная валидация top-50 протоколов.
- [ ] Настоящий gold set.
- [ ] Полная база дозировок.
- [ ] Переключение production gate на v3.
- [ ] Массовый пересчёт всех MIS КЗ.

P2 требует методистов, внешних данных или отдельного rollout и не должен имитироваться.

---

## 19. Итоговый отчёт

Создать:

```text
docs/reports/kz-evaluation-v3-overnight-result-2026-07-28.md
```

Содержание:

1. SHA baseline.
2. Ветка.
3. Что реализовано.
4. Какие файлы изменены.
5. Архитектура v3.
6. Правила trust.
7. Изменение structural score.
8. Coverage/confidence semantics.
9. Corpus audit metrics.
10. Shadow benchmark.
11. Тесты и длительность.
12. Известные ограничения.
13. Что требует методиста.
14. Команда продолжения.
15. Commit SHA и remote branch.

Не заявлять улучшение клинической точности без gold-метрик. Допустимые формулировки:

- «устранён архитектурный источник ложных штрафов»;
- «недоверенные правила переведены в advisory»;
- «добавлено измерение coverage»;
- «подготовлен shadow scorer».

---

## 20. Commit и push

### 20.1. Перед коммитом

```bash
git status --short
git diff --check
git diff --stat
git diff
```

Проверить:

- нет `.env`;
- нет ключей;
- нет ПДн;
- нет PDF пациентов;
- нет runtime caches;
- нет огромных generated JSONL;
- нет случайных изменений вне задачи.

### 20.2. Коммиты

Предпочтительно 2–4 осмысленных коммита:

```text
feat(kz): add trust-aware evaluation v3 contract
fix(kz): make documentation and coverage scoring explicit
feat(protocols): add knowledge audit and trusted requirements
test(kz): add v3 shadow and safety regressions
```

Допустим один итоговый коммит, если разделение создаёт риск потерять согласованность.

### 20.3. Push

```bash
git push -u origin codex/kz-evaluation-quality-v3
```

Не push в `main`. Не force-push.

Если push не проходит из-за сети/авторизации:

- повторить после проверки remote;
- не менять credentials;
- сохранить готовые коммиты;
- записать точную ошибку и команду повторения в итоговый отчёт.

---

## 21. Финальный ответ Cursor

После успешного push сообщить только проверяемые факты:

- что сделано;
- какие главные архитектурные дефекты устранены;
- результаты тестов;
- corpus/shadow метрики;
- commit SHA;
- имя ветки;
- ссылка на remote branch/PR, если создан;
- что осталось только из-за необходимости methodist gold/внешних данных.

Не завершать с формулировкой «план готов». Задача считается выполненной только после
реализации P0, тестов, отчёта, коммита и push.

