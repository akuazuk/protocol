# Итог ночной итерации: scorer v3 КЗ (trust-aware, coverage, shadow)

**Дата:** 2026-07-28
**ТЗ:** `docs/plans/2026-07-27-kz-evaluation-quality-overnight-v1.md`
**Режим:** автономная сессия, отдельный clean worktree + ветка.

## 1. Baseline
- SHA baseline: `ce82af2` (main на момент старта).
- Ветка задачи: `codex/kz-evaluation-quality-v3` (worktree `/private/tmp/protocol-kz-evaluation-v3`).
- Python: 3.12.3.
- BUILD_VERSION до: `2026-07-27-r16-miskz-fix-js-vivid-restyle` → после: `2026-07-27-r17-kz-evaluation-v3-shadow`.
- Scorer/schema: новый контракт `schema_version=3.0`, `scorer_version=2026-07-27.1`.
- Целевые тесты до изменений: `test_scoring`, `test_compliance_gate`, `test_consult_alignment`,
  `test_kz_deep_eval`, `test_consult_tiering` — зелёные (проверено в clean worktree).

## 2. Ветка
`codex/kz-evaluation-quality-v3`, push в `origin` (см. §15).

## 3. Что реализовано
- **Единый контракт `KzEvaluationResultV3`** (Pydantic): `score`, `axes`, `coverage`,
  `confidence`, `risk`, `protocols`, `findings`, `diagnostics`, `mode`, `provenance`,
  `legacy`. Безопасные defaults, без NaN/Inf, score 0-100, coverage/confidence 0-1.
- **Trust levels A-D** для правил (`rule_trust.py`): консервативный mapping из
  rule_source/review/quote; A/B штрафуют только с подтверждённой цитатой; C — advisory;
  D — heuristic. C/D не штрафуют и не гейтят.
- **Coverage-aware structural score** (`kz_evaluation_engine.score_documentation`):
  обязательные поля НЕ компенсируются рекомендуемыми (раздельные completion + явные cap:
  нет диагноза → 45, нет рекомендаций → 55, нет объективного статуса на первичном → 65,
  пустое КЗ → insufficient_data). `None`-блоки отражаются в `coverage`, а не исчезают.
- **Единый scorer поверх осей A/B/C/D** с risk-gate: подтверждённый P0 (trust A/B) → hard
  cap 40 + critical; P1 → cap 60; coverage-band (limited/insufficient evidence);
  confidence-band (низкая уверенность → review). Confidence отделён от клинического score.
- **Protocol applicability** (`kz_protocol_applicability.py`): раздельные уверенности +
  жёсткие инварианты (детский КП не штрафует взрослое КЗ; стационарный КП не штрафует
  амбулаторное; fallback/generic/rehab → advisory; `applicability_confidence < 0.75` →
  advisory-only).
- **Каноническая knowledge-model протокола** (`protocol_knowledge_model.py`):
  `ProtocolKnowledgeDocument`/`AtomicRequirement`/…, адаптер `ProtocolSummary → knowledge`
  (auto → C, path/rich-table → D), валидатор penalty-ready + CLI.
- **Аудит корпуса + очередь методиста** (`scripts/audit_kz_protocol_knowledge.py`).
- **Shadow benchmark** (`scripts/compare_kz_evaluation_v3.py`) на синтетических фикстурах.
- **Trust-aware medication findings** (`medication_findings.py`).
- **Gold-annotation инфраструктура** (`kz_gold_annotation.py`): схема, sample builder,
  двойная разметка/арбитраж, evaluator QWK/MAE/harm-recall/false-critical, синтетика.
- **API аддитивно**: батч `run_mis_protocol_l1_batch.py` пишет `evaluation_v3` в shadow;
  дашборд-деталь отдаёт `evaluation_v3` (feature-flag методиста). Legacy-поля не тронуты.

## 4. Изменённые/новые файлы
Новые: `clinical_knowledge/kz_evaluation_schema.py`, `kz_evaluation_engine.py`,
`rule_trust.py`, `kz_protocol_applicability.py`, `protocol_knowledge_model.py`,
`medication_findings.py`, `kz_gold_annotation.py`;
`scripts/audit_kz_protocol_knowledge.py`, `scripts/compare_kz_evaluation_v3.py`;
`tests/test_kz_evaluation_v3.py`, `test_rule_trust.py`, `test_kz_coverage_scoring.py`,
`test_kz_v3_gate.py`, `test_protocol_knowledge_model.py`, `test_protocol_knowledge_audit.py`,
`test_medication_findings.py`, `test_kz_gold_annotation.py`;
`tests/fixtures/kz_v3_cases.jsonl`; компактные отчёты в `data/ml/reports/`.
Изменены (аддитивно): `scripts/run_mis_protocol_l1_batch.py`,
`clinical_knowledge/mis_kz_quality.py`, `rag_server.py` (BUILD_VERSION).

## 5. Архитектура v3
КЗ → документирование (coverage-aware) + concordance (trust/applicability-aware) +
safety (curated → penalty; protocol red flags → trust-aware) + regulatory (№55) →
взвешенное среднее осей → risk-gate → coverage/confidence-band → canonical result.
Shadow: `KZ_EVALUATION_V3_ENABLED=1`, `KZ_EVALUATION_V3_PRIMARY=0`, `KZ_EVALUATION_V3_GATE=0`.

## 6. Правила trust
A=approved методистом; B=reviewed+подтверждённая цитата; C=auto/summary без review;
D=path/rich-table/fallback. Penalty только A/B c цитатой. C/D — advisory (needs_human),
снижают coverage/confidence, не штрафуют, не блокируют gate.

## 7. Изменение structural score
Раздельные required/conditional/recommended; формула `0.70·required + 0.20·conditional +
0.10·recommended` c ренормализацией по применимым группам; явные cap вместо косвенного
штрафа; пустое КЗ → insufficient_data.

## 8. Coverage/confidence semantics
`coverage` — доля потенциально применимых проверок, реально выполненных доверенными
данными (concordance без trusted-протокола → низкое покрытие). `confidence` —
document_parse / protocol_match / evidence_match / protocol_knowledge; влияет на статус,
gate и необходимость ревью, но НЕ на клинический балл.

## 9. Corpus audit metrics (факт)
- Протоколов: **477**; атомарных требований: **11 259**.
- source_verified_coverage_pct: **99.0%** (цитаты есть почти везде).
- penalty_eligible_coverage_pct: **0.0%**; methodist_approved_coverage_pct: **0.0%**;
  protocols_without_safe_penalty_rule: **477**.
- Вывод: «наличие правила ≠ пригодность к штрафу» подтверждено количественно — без
  методистского review весь корпус advisory. Очередь методиста сформирована по приоритету.

## 10. Shadow benchmark (синтетические фикстуры, N=10)
- legacy deep score mean **62.7** / median 63.1; v3 score mean **83.2** / median 89.4.
- caps применён: 1; смен статуса: 8; C/D findings исключены из штрафа: **4**;
  протокол advisory (не penalty-eligible): **2** (детский/стационарный КП).
- Разница средних отражает устранение ложных штрафов по недоверенным протоколам, а не
  «улучшение клинической точности» (gold-метрик пока нет).

## 11. Тесты и длительность
- Узкие + новые: `test_scoring test_compliance_gate test_consult_alignment test_kz_deep_eval
  test_consult_tiering test_rule_trust test_kz_coverage_scoring test_kz_evaluation_v3
  test_kz_v3_gate test_protocol_knowledge_model test_medication_findings
  test_kz_gold_annotation` — **зелёные** (~36 c).
- `test_protocol_knowledge_audit` — зелёный (~60 c, грузит корпус).
- Полный `pytest` — результат см. §11 в чек-листе ТЗ и вывод сессии.
- Ruff по изменённым файлам — зелёный.

## 12. Известные ограничения
- Без методистского gold нельзя заявлять рост клинической точности — заявляем только
  устранение архитектурного источника ложных штрафов и добавление coverage.
- Concordance без протокола опирается на базовые проверки (dx-support, форма МКБ);
  протокольное покрытие advisory до методистского review.
- `evaluation_v3` в UI — только data-поле (feature-flag), без визуального редизайна.

## 13. Что требует методиста / внешних данных (P2)
- Ручная валидация top-протоколов из очереди (→ trust B/A → penalty-eligible).
- Настоящий gold set (800-1200 КЗ) и калибровка порогов/весов.
- Полная база дозировок; включение production gate на v3.

## 14. Команда продолжения
```bash
# аудит корпуса и очередь методиста
python -m scripts.audit_kz_protocol_knowledge
# shadow benchmark
python -m scripts.compare_kz_evaluation_v3
# массовый v3 в батче (shadow) - на машине с БД:
KZ_EVALUATION_V3_ENABLED=1 python -m scripts.run_mis_protocol_l1_batch --deep-eval ...
# валидатор knowledge-model одного протокола
python -m clinical_knowledge.protocol_knowledge_model --validate <summary.json>
```

## 15. Commit SHA и remote branch
- Ветка: `codex/kz-evaluation-quality-v3` (push в `origin` выполнен).
- Коммиты поверх baseline `ce82af2`:
  - `ab5af54` feat(kz): trust-aware evaluation v3 contract, engine и applicability
  - `b7244ee` feat(protocols): knowledge-model + аудит корпуса + очередь методиста
  - `f6bd0de` feat(kz): medication findings, gold, shadow benchmark + аддитивный API
- Полный pytest: **6 pre-existing baseline failures** (подтверждены на baseline main
  worktree, модули не затронуты задачей): `test_drug_normalizer` (amoxicillin
  нормализация ×2), `test_consult_cache::test_same_pdf_returns_identical_result`,
  `test_medication_safety::test_obgyn_61_...pregnancy`,
  `test_assist_search_speed` (×2). Остальные — зелёные. Новые v3-тесты (44) зелёные.
