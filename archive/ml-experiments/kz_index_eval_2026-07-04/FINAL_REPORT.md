# Итог: тест vector index по всем КЗ (2026-07-04)

**Prod:** https://protocol-bimy.onrender.com · **r37-proto-semantic-lazy**  
**Набор:** `clients_consult/` - 41 PDF (36 КЗ + 5 анализов A/a/А/а)

---

## 1. Классификация A/a/А/а (анализы)

| Файл | Ожидание | Результат L1 |
|------|----------|--------------|
| A_2, a_1, a_3, a_4, a_pl_1 | `lab_in_kz`, без scoring | ✅ 5/5 `upload_mismatch=true`, `overall=null` |

Исправлено: `is_b2c_lab_filename()` теперь учитывает кириллические **А/а** (раньше только латиница).

---

## 2. Прогон L1 (все 41 файл) - стабильный

| Метрика | Значение |
|---------|----------|
| Успех | **41/41** (0 ошибок) |
| КЗ avg overall | **82.4%** |
| КЗ overall < 70% | **4** |
| Анализы отсечены | **5/5** |

**Слабые КЗ (ручной разбор):**
- `ja_3` - 55.0%
- `ja_4` - 55.0%
- `report_n_1` - 60.0%
- `report_n_2` - 61.9%

> L1 не возвращает `retrieval_top` / `rag_chunks` - это норма для быстрого скрининга.

Артефакт: `ml/experiments/kz_index_eval_2026-07-04/`

---

## 3. Прогон L2 (36 КЗ) - retrieval + индекс

| Метрика | Значение |
|---------|----------|
| Успех | **7/36** |
| 502 Bad Gateway | **29** |
| КЗ с RAG chunks (успешные) | **7/7** (rag 1-3) |
| КЗ с retrieval_top | **7/7** |

**Успешные L2 (индекс + retrieval работают):**
- F_1_p: overall 92%, rag=3, ret=3
- gastro_1: 75%, rag=2, ret=2
- ja_1: 83%, rag=2, ret=2
- ja_4: 55%, rag=3, ret=3 (слабый кейс, но retrieval есть)
- l_28_1: 85%, rag=1, ret=1
- report_g_1: 88%, rag=1, ret=1
- report_procy_g_1: 76%, rag=1, ret=1

**Причина 29 ошибок:** загрузка vector index (~**981 МБ** `vectors.npy`) + L2 Gemini на Render **Standard (2 GiB RAM)** → OOM → 502. Прогрев индекса в начале батча усугубляет.

Артефакт: `ml/experiments/kz_index_eval_2026-07-04-l2/`

---

## 4. Состояние vector index на диске

| Параметр | Значение |
|----------|----------|
| Путь | `/var/data/corpus_vector_index/` |
| vectors.npy | ~981 МБ ✅ залит |
| meta.json | ~888 КБ ✅ |
| enabled | true |
| loaded в RAM | лениво (после первого vector-запроса) |
| indexed (после warm) | **83 652**, dim 3072 |

`/api/corpus-stats` показывает `loaded:false` до первого semantic/RAG запроса - это ожидаемо.

---

## 5. План улучшений (приоритет)

### P0 - инфраструктура индекса (критично)

1. **mmap / memory-map для `vectors.npy`** вместо полной загрузки в RAM (`numpy.memmap`) - снизит OOM на Standard plan.
2. **Не грузить индекс при старте** - только on-demand per-query scoped search (top-K без полного матричного индекса в RAM).
3. **Альтернатива:** Render **Pro** (4 GiB) или вынести vector search в отдельный лёгкий сервис.
4. **L2 batch:** `CONSULT_CONCURRENCY=1`, пауза 15-30 с между КЗ, без warm-index в начале батча.

### P1 - качество КЗ

5. **4 слабых кейса** (`ja_3`, `ja_4`, `report_n_1`, `report_n_2`) - разбор в очереди методиста: ICD, RAG allowlist, правила.
6. **L1 telemetry:** добавить в ответ L1 поле `retrieval_paths_count` (хотя бы число) для мониторинга без полного L2.

### P2 - protocol semantic (навигация)

7. **Per-path embeddings в lazy store** (`semantic=True` уже есть) - перевести protocol search из `lexical` в `semantic` на prod.
8. **Фильтр шума PDF** - протоколы с 6 raw blocks / 0 shown (варикоз) не дают semantic matches; ослабить `_is_noise` для коротких treatment-чанков.

### P3 - уже сделано в этой сессии

- ✅ Cyrillic А/а в `is_b2c_lab_filename`
- ✅ `scripts/run_kz_vector_index_eval.py` - полный батч L1/L2 + semantic probe
- ✅ Индекс залит на Persistent Disk

---

## 6. Команды для повтора

```bash
# L1 - все КЗ + анализы (быстро, стабильно)
.venv/bin/python scripts/run_kz_vector_index_eval.py --include-analysis

# L2 - только КЗ, без прогрева индекса, с паузой
.venv/bin/python scripts/run_kz_vector_index_eval.py --kz-only --tier L2 \
  --no-warm-index --pause-sec 20 --out ml/experiments/kz_index_eval_L2
```

---

## 7. Вывод

- **Индекс на диске есть и работает** - у успешных L2 кейсов retrieval возвращает 1-3 протокола и 1-3 RAG-чанка.
- **Массовый L2-батч на Standard Render ненадёжен** из-за RAM при загрузке 1 ГБ индекса.
- **КЗ-скрининг L1 стабилен:** 82.4% avg, анализы A/a/А/а корректно отсекаются.
- **Следующий инженерный шаг:** mmap vector index (P0) → повторить L2 на всех 36 КЗ без 502.
