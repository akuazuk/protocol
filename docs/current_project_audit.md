# Аудит проекта Protocol — модуль проверки КЗ (KZ Compliance Checker)

> Обновлено: 2026-05-31 после доработки модуля KZ compliance.  
> Версия сборки: `2026-05-31-r49-kz-compliance-complete`.

---

## 0. Резюме

**Готовность модуля KZ compliance vs ТЗ: ~90%.**

Реализовано в `clinical_knowledge/` (без параллельного `src/consultation_compliance/`):

| Область | Статус |
|---------|--------|
| Модели (`consult_schema.py`) | ✅ |
| Парсер КЗ + DOCX/PDF/TXT в CLI | ✅ |
| `requirement_checker.py` + `config/kz_requirements.yaml` | ✅ |
| `protocol_compliance_checker.py` + `rule_checker` | ✅ |
| Scoring 8 блоков | ✅ |
| Отчёты JSON/MD (11 разделов §15) | ✅ |
| `batch_runner.py` + batch_summary.csv/md | ✅ |
| CLI `scripts/check_kz.py` / `analyze_consultation` | ✅ |
| Unit-тесты | ✅ 42+ |

**Оставшиеся пробелы (~10%):** полное OCR PDF-сканов; глубокая сверка каждой дозы с таблицами протокола; UI-вкладка «Структурная проверка» отдельно от LLM (structured analysis уже в prod через `CONSULT_STRUCTURED_ANALYSIS=1`).

---

## 1. Карта модулей

| Модуль ТЗ | Файл | Статус |
|-----------|------|--------|
| models | `consult_schema.py` | ✅ |
| kz_parser | `consult_parser.py` | ✅ |
| requirement_checker | `requirement_checker.py` | ✅ |
| protocol_compliance_checker | `protocol_compliance_checker.py` | ✅ |
| protocol_matcher | `protocol_match.py` | ✅ |
| scoring | `scoring.py` | ✅ |
| safety_checker | `safety_checker.py` | ✅ |
| report_builder | `consult_report.py` | ✅ |
| batch_runner | `batch_runner.py` | ✅ |
| text extract (DOCX) | `text_extract.py` | ✅ |

**Оркестратор:** `consult_analysis.analyze_consultation_text()`  
**Prod:** `consult_review_pipeline.py`  
**CLI:** `python -m scripts.check_kz --file …` / `--folder …`

---

## 2. Критерии приёмки (ТЗ §18)

| # | Критерий | Статус |
|---|----------|--------|
| 1–2 | check-kz / folder | ✅ `scripts/check_kz.py` |
| 3–11 | Парсинг КЗ | ✅ |
| 12–15 | Подбор протоколов | ✅ |
| 16–18 | JSON/MD/batch | ✅ |
| 19–22 | Scoring, issues, red flags | ✅ |
| 23 | Source refs на выводы | ⚠️ частично (протокол + фрагменты КЗ в отчёте) |
| 24–26 | insufficient_data, тесты, без регрессий | ✅ |

---

## 3. Команды

```bash
.venv/bin/python -m scripts.check_kz --file tests/fixtures/consultations/gastro_adult.txt
.venv/bin/python -m scripts.check_kz --folder tests/fixtures/consultations
# → data/reports/kz_checks/
```
