# План улучшения проверки КЗ по протоколам РБ

> По `docs/cursor_task_improve_kz_protocol_compliance.md`, 2026-05-31.  
> Принцип: **расширять** `clinical_knowledge/`, не переписывать RAG/UI/API.

---

## 1. Текущая архитектура (что уже есть)

| Слой | Модули | Назначение |
|------|--------|------------|
| Корпус | `download_minzdrav_protocols.py`, `corpus_pipeline/` | PDF → chunks, cards, rules |
| RAG | `rag_server.retrieve()` | Лексика + BM25 + embed-rerank |
| Prod pipeline | `consult_review_pipeline.py` | LLM + RAG + structured analysis |
| Парсер КЗ | `consult_parser.py`, `diagnosis_parser.py`, `medication_parser.py`, `template_parser.py` | ConsultationDocument |
| Структура | `requirement_checker.py`, `config/kz_requirements.yaml` | Required/conditional rubrics |
| Протоколы | `protocol_match.py`, `applicability.py` | Карточки + applicability |
| Правила | `rule_checker.py`, `protocol_compliance_checker.py` | Deterministic rules |
| Оценка | `compliance_engine.py`, `scoring.py`, `safety_checker.py` | ComplianceReport |
| Гибрид UI | `consult_overall_score.py` | 80% structured + 20% rules; alignment-критерии детерминированы |
| CLI | `scripts/check_kz.py`, `batch_runner.py` | Batch JSON/MD/CSV |
| Отчёты | `consult_report.py` | JSON / Markdown / HTML |

**Три слоя оценки (целевая модель ТЗ):**

- **A.** Формальная полнота → `requirement_checker` + `documentation_score`
- **B.** Применимость протокола → `protocol_match` + `protocol_applicability_score`
- **C.** Соответствие протоколу → `rule_checker` + `evidence_map` + exam/treatment assessments

---

## 2. Пробелы vs ТЗ (обновлено 2026-06-01)

| Требование | Статус |
|------------|--------|
| Детерминированные критерии КЗ (МКБ / КП / НПА) | **Готово** — `consult_alignment.py`, L1+L2 |
| Rich rules из table-чанков | **Готово** — `rich_rules_supplement.py`, runtime + catalog build |
| Typed retrieve (diagnostics/treatment/monitoring) | **Готово** — `supplement_retrieval_from_rich_chunks` |
| Offline индекс профилей КП | **Готово** — `protocol_icd_profile_index.py`, `scripts/build_protocol_icd_index.py` |
| НПА наблюдение (№127) | **Базово** — `data/regulations/mz_2015_127.json` |
| Связь 8 блоков ↔ alignment | **Готово** — `alignment_by_block` в compliance |
| Evidence из alignment | **Готово** — `append_alignment_evidence` |
| Пересборка `data/catalog/rules` на prod | **Нужен deploy** — `python -m scripts.build_catalog_rules` на Render |

### Исторические пробелы (закрыты ранее)

---

## 3. Новые / расширяемые файлы

| Файл | Действие |
|------|----------|
| `clinical_knowledge/rule_model.py` | **Новый** - ProtocolRule, RuleApplicability |
| `clinical_knowledge/evidence_map.py` | **Новый** - построение evidence map |
| `clinical_knowledge/table_rule_extractor.py` | **Новый** - правила из table_block чанков |
| `clinical_knowledge/confidence_scoring.py` | **Новый** - confidence_score |
| `clinical_knowledge/consult_schema.py` | Расширение моделей (backward-compatible) |
| `clinical_knowledge/diagnosis_parser.py` | safety_flags, red_flag_finding |
| `clinical_knowledge/template_parser.py` | source_text в TemplateBlock |
| `clinical_knowledge/protocol_match.py` | взвешенный score, per-diagnosis |
| `clinical_knowledge/requirement_checker.py` | контекст TZ §6 |
| `clinical_knowledge/rule_checker.py` | applicability 2.0, type mapping |
| `clinical_knowledge/compliance_engine.py` | evidence map, confidence, caps |
| `clinical_knowledge/safety_checker.py` | required_actions, cap_if_unhandled |
| `config/red_flags.yaml` | cap_if_unhandled, drug_safety |
| `config/compliance_weights.yaml` | documentation_score alias weights |
| `clinical_knowledge/consult_report.py` | JSON/MD §17-18 |
| `clinical_knowledge/batch_runner.py` | расширенный CSV |
| `tests/test_regression_kz_compliance.py` | **Новый** |

**Не трогаем:** `rag_server.py` (кроме BUILD_VERSION), `index.html`, контракт `/api/consult-review` (только additive JSON fields).

---

## 4. Обратная совместимость

1. **ScoreBreakdown:** сохраняем `structural_score`, `protocol_match_score`; дублируем в `documentation_score`, `protocol_applicability_score`.
2. **OverallStatus:** старые статусы без изменения семантики; новые - только при условиях ТЗ.
3. **API:** новые поля в `structured_analysis.compliance` - опциональные; UI читает `overall_compliance_pct` из hybrid scorer.
4. **CLI:** те же команды; расширенный batch CSV.
5. **Env:** все существующие флаги (`CONSULT_STRUCTURED_ANALYSIS`, `CONSULT_OVERALL_HYBRID`) без изменений.

---

## 5. Этапы реализации

| # | Этап | Ключевые deliverables |
|---|------|----------------------|
| 1 | Parser + evidence map | schema, diagnosis flags, evidence_map builder |
| 2 | requirement_checker | контекст first_visit, has_red_flag, evidence_fields |
| 3 | protocol_matcher | weighted score, per-diagnosis, not_applicable list |
| 4 | rule model 2.0 | rule_model.py, rule_checker adapter |
| 5 | table_rule_extractor | extract from table chunks JSONL |
| 6 | compliance_engine | exam logic, evidence_map integration |
| 7 | safety_checker | partially_handled, cap_if_unhandled |
| 8 | scoring + confidence | confidence_scoring, new statuses |
| 9 | reports | JSON/MD/batch |
| 10 | regression tests | test_regression_kz_compliance.py |

---

## 6. Риски

| Риск | Митигация |
|------|-----------|
| Поломка тестов scoring | Alias полей + перенормировка весов |
| UI ожидает старые ключи JSON | Additive-only fields |
| Rule catalog не в формате 2.0 | Adapter: legacy dict → ProtocolRule |
| Нет полных fixture PDF | Unit-тесты на текстовых сниппетах из ТЗ §16 |

---

## 7. Критерии готовности

- [ ] 42+ существующих тестов проходят
- [ ] Новые regression-тесты §16 (сниппеты)
- [ ] JSON содержит `confidence_score`, `evidence_map`, `score_source`
- [ ] LLM не повышает итоговый % (`consult_overall_score`)
- [ ] `docs/improvement_plan.md` актуален
