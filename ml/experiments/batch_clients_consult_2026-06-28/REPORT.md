# Batch clients_consult — 2026-06-28 (после chunk-gemini-final)

**Папка:** `clients_consult/` — **29 PDF**  
**Сервер:** https://protocol-bimy.onrender.com · `2026-06-28-r66-chunk-gemini-final`  
**Скрипт:** `scripts/run_clients_consult_render_batch.py --ai-review auto`

## Сводка L1 (Render)

| Метрика | Значение |
|---------|----------|
| Прогон | **29/29 OK** (~28 с на весь batch) |
| Средний overall | **81.1%** |
| overall &lt; 70% | **6 кейсов** |
| overall ≥ 85% | **13 кейсов** |
| failed rules (L1) | **0** (rules_check на Render L1 не отдаёт findings) |
| rag_used (L1) | **False** на всех — L1 = structured + alignment, без chunk-RAG |

## Распределение

| Группа | case_id | overall % | Комментарий |
|--------|---------|-----------|-------------|
| **Низкие &lt;70** | report_n_1 | 60.0 | treatment 25%, safety 0%, critical=3 |
| | a_1, a_3, a_4, report_n_2 | 61.9 | status `low_confidence` / sparse |
| | A_2 | 69.9 | documentation 12.5% |
| **Средние 70–85** | gastro_1, report_procy_g_1, pl_*, report_aler_1, … | 75–84 | cap / частичное соответствие |
| **Высокие ≥85** | kard_1, F_1_p, mg_1, report_lor_jarov_1, report_urolog_1, … | 85–93 | стабильно |

## Сравнение с batch 2026-06-14 (локальный L1 + rules)

| case_id | Jun-14 | Jun-28 Render | Δ |
|---------|--------|---------------|---|
| report_lor_1 | 55.0 | **88.5** | **+33.5** |
| kard_1 | 89.4 | 92.4 | +3.0 |
| report_ter_1 | 88.6 | 88.7 | ≈0 |
| gastro_1 | 75.0 | 75.0 | 0 |

**report_lor_1** — главный выигрыш (вероятно r118+ caps + деплой; chunk corpus косвенно через alignment).

## Что видно по блокам (проба 4 кейса)

| case | documentation | protocol | diagnosis | treatment | safety |
|------|---------------|----------|-----------|-----------|--------|
| A_2 | **12.5** | 90 | 92 | — | 100 |
| report_n_1 | 100 | 90 | 92 | **25** | **0** |
| kard_1 | 100 | 90 | 92 | 80 | 100 |
| report_ter_1 | 100 | 90 | 70 | — | 100 |

Повторяющийся паттерн: **protocol_applicability ≈ 90%** на всех — возможно слишком оптимистичный дефолт.

## AI-review

Auto AI-review на Render вернул **422** (нет `analysis_id` в ответе tier на prod). Нужен фикс autolog snapshot или отдельный вызов с сохранением id.

---

## Рекомендации (приоритет)

### A. Движок КЗ (быстрый эффект, без ML)

1. **Sparse / short KZ** (`a_*`, `report_n_2`) — единый gate `low_confidence` при text_len &lt; 1500 и слабом OCR; не давать 61.9% как «финальный» балл без предупреждения.
2. **report_n_1** — разобрать treatment/safety блоки (critical=3); вероятно ложные негативы по отсутствующим полям в коротком шаблоне.
3. **A_2** — documentation_score 12.5 при хорошем diagnosis — калибровка весов блоков.
4. **Включить rules_check в L1 на Render** (сейчас `rules_pct: null`) — иначе batch не ловит регрессии правил.
5. **protocol_applicability 90%** — пересмотреть cap/дефолт; слишком мало дискриминации между кейсами.

### B. Второй прогон Gemini chunk QA (целевой)

Не «ещё 12k с нуля», а **точечная очередь ~500–2000**:

| Источник | Зачем |
|----------|--------|
| `retrieval_fix` из methodist feedback | протокол hit@3 |
| Чанки протоколов из **6 слабых KZ** (nevrologiya, ter, lor, a_*) | RAG для L2 |
| 32 пропущенных chunk_id | закрыть хвост |
| `chunk_qa_review.jsonl` (9 шт.) | ручная валидация |

После merge — **upload Render + restart** (как сегодня).

### C. Локальное обучение (средний горизонт)

**Не учить «на метках чанков» напрямую** — мало связи с баллом КЗ. Лучше:

| Датасет | Модель / артеfact | Эффект |
|---------|-------------------|--------|
| `retrieval_pairs` из batch + methodist `retrieval_fix` | **LoRA embedder** (`finetune_embedder.py`) | правильный протокол в top-3 |
| `analysis_review` overrides (rule_pass human vs system) | **rule family gates** (код, не LLM) | меньше false positive |
| 29 KZ × score_breakdown | **калибратор блоков** (логрег/GBM на фичах) | точнее overall |
| Chunk `verdict` + `entities` | только как **weak labels** для reranker features | вторично |

Минимум для старта ML: **50+ retrieval_fix** с `chosen_path` / `rejected_path` (см. `docs/ml-backlog-when-kz-ready.md`).

### D. Следующий batch-тест

```bash
# L1 все KZ на Render (повтор):
.venv/bin/python scripts/run_clients_consult_render_batch.py --ai-review auto

# L2 на 6 слабых (проверка chunk-RAG):
for c in report_n_1 a_1 A_2 report_n_2 gastro_1 report_procy_g_1; do
  curl -s -X POST .../api/consult-review/tier -d "{\"tier\":\"L2\",\"text\":\"...\"}"
done
```

---

## Файлы

- `report.json`, `batch_summary.csv` — полный прогон
- `scripts/run_clients_consult_render_batch.py` — повторяемый runner
