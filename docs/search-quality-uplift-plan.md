# План кратного улучшения качества поиска по протоколам

Статус: черновик к реализации. Автор: агент. Дата: 2026-07-20.

Цель (из запроса): поиск по протоколам должен выдавать **только точные результаты** без
обрывков текста и без бессмысленности; препараты и обследования - только правильные и
относящиеся к протоколу, **с оценкой вероятности**; в выдаче должно быть понятно, протокол
**для стационарного или амбулаторного** лечения. Плюс - разобраться в структуре самих протоколов.

Этот документ - карта архитектуры «как есть», разбор корневых причин проблем и
поэтапный план (треки 1-4) с критериями приёмки и оценкой рисков.

---

## 1. Архитектура поиска «как есть»

### 1.1 Уровни поиска

Запрос из UI (`index.html`, `runAssistSearch`) идёт в `POST /api/assist` (`rag_server.py`).
Уровень выбирается через `clinical_knowledge/search_tiering.py`:

| Уровень | Что делает | LLM |
|---------|-----------|-----|
| S0 | Только МКБ через `clinical_knowledge/protocol_icd_index.py` (без retrieve) | нет |
| S1 (по умолчанию) | `retrieve()` лексика + BM25 + вектор, `retrieve_only=true` | нет |
| S2 | S1 + Gemini для ранжирования и сводки | да |

Есть также воронка `POST /api/search/funnel` (`clinical_knowledge/search_funnel.py`),
шаги: длина запроса -> аудитория -> МКБ -> рубрики -> протоколы -> навигация по секциям.

### 1.2 Пайплайн одного запроса

1. `clinical_query_for_rag` (`rag_server.py`) - вырезает блок «Жалобы и вопрос», убирает ответы на уточнения.
2. `_infer_icd_pipeline_from_full_query` - МКБ через `icd_mkb.py` (`analyze_query_for_icd`, `finalize_icd_analysis_codes`), затем `expand_query_for_retrieve` (`clinical_knowledge/search_query_expand.py`).
3. `build_protocol_search_context` (`clinical_knowledge/search_retrieval.py`) - расширение МКБ, `path_boost`, `path_allowlist`.
4. `retrieve()` / `_retrieve_core` - генерация кандидатов и скоринг.
5. Формирование excerpt-ов и confidence.
6. Опционально `extract_clinical_detail` (LLM) для блока «На приёме».

### 1.3 Ядро retrieve()

- Кандидаты: токены (`tokenize_ru`) + токены МКБ-лексикона + вектор-префильтр (FAISS, `clinical_knowledge/vector_index.py`, при `RAG_VECTOR_INDEX=1`).
- Ленивый режим (Render/manifest): `clinical_knowledge/chunk_store.py`, `lazy_rag_config.py`. Без allowlist - retrieve возвращает пусто (`forbid_full_corpus_retrieve`).
- Скоринг: лексические веса + BM25 (`retrieval_bm25.py`, `RAG_LEX_BM25_ALPHA`) + маршрутизация (`RAG_ROUTING`) + ICD-boost + audience-фильтр + boost по типу чанка/summary.
- Реранк эмбеддингами: `RAG_GEMINI_EMBED_RERANK` или `RAG_PRECOMPUTED_CHUNK_EMBED`. На проде часто **выключен**.
- Вывод: `RAG_MAX_CHUNKS` (6), `max_per_path` (2), excerpt через `format_excerpt_for_display` (`RAG_EXCERPT_CHARS`, 700).

### 1.4 Структура протоколов (6 слоёв)

| Слой | Файлы | Что даёт |
|------|-------|----------|
| Сырые чанки | `corpus_chunks_parts/*.jsonl` (104 687, разбивка `_L*`) | text, chunk_type, page, icd, sparse drugs/care_setting |
| Rich-чанки | `output/rich_chunks/rich_chunks.jsonl` (63 775, 1 doc_id/PDF) | теги care_setting/obligation/entities, icd10_weights |
| Summary Cards | `data/protocol_summaries/yaml|json` (~478) | conditions[] -> exams/treatment/criteria/red_flags **с source_ref.quote и страницей** |
| Summary RAG-срезы | `data/protocol_summaries/summary_chunks.jsonl` (355) | секции с цитатами (grounded) |
| Каталог | `index.csv`, `data/protocol_catalog.jsonl` (478) | audience, icd10_primary, icd10_weights, protocol_kind |
| ICD-профили | `data/catalog/protocol_icd_profiles.jsonl` (478) | diagnostics/medications/treatment с `obligation: required|recommended` + `cites -> chunk_id` |

Ключевой вывод: **самый качественный, привязанный к цитатам слой (Summary Cards и
`protocol_icd_profiles`) уже существует**, но в основном выводе поиска почти не используется,
а на проде частично выключен ради экономии RAM.

---

## 2. Корневые причины проблем

### 2.1 Обрывки текста
- `gather_protocol_text` (`rag_server.py`) режет чанк посередине: `out.append(t[:rest])`.
- `RAG_EXTRACT_ITEM_MAX_CHARS=420`, UI `excerptPreviewLen` 140-200 - добавляют `…` посреди фразы.
- Границы PDF-чанков рваные (переносы, дефисы переноса).
- Готовое sentence-aware решение `clinical_knowledge/meaningful_excerpt.py` используется только в КЗ, не в поиске.

### 2.2 Бессмысленность и неправильные препараты/обследования
- Нет пер-пунктового grounding: LLM возвращает `medications[]` / `investigations[]`, и **нет проверки**, что пункт реально есть в тексте протокола.
- Механизмы проверки есть, но применяются в другом месте: `verify_protocol_excerpt` (`consult_criteria_enrichment.py`), `quote_found_in_source` (`protocol_summary/quote_validator.py`).
- `SYSTEM_EXTRACT_NON_PROTOCOL` дописывает общеклинические пункты `[не из протокола]` (по умолчанию включено).
- `_fallback_algorithms_from_ext` синтезирует шаги по regex.
- Lite-режим (`protocol_practical_lite.py`) выбирает строки по regex, при промахе - `lines[:4]`.

### 2.3 Отключён лучший слой на проде (компромисс по RAM)
В `_apply_low_memory_defaults` (`rag_server.py`, профили 512 MiB и 2 GiB) на Render принудительно:
- `PROTOCOL_SUMMARY_RAG_MERGE=0` - слой Summary-срезов с цитатами выключен;
- `RAG_GEMINI_EMBED_RERANK=0` - семантический реранк выключен;
- `RAG_LEX_BM25_ALPHA=1.0` - фактически без BM25-blend.

Это осознанная экономия памяти, а не баг. Значит любое включение - это trade-off по RAM,
и нужно либо поднять план Render, либо использовать лёгкие альтернативы (precomputed embeddings,
маленький summary-merge на 355 строк почти бесплатен по памяти).

### 2.4 Стационар vs амбулаторно почти не работает
- В `index.csv` и `protocol_catalog.jsonl` **нет поля care_setting** на уровне PDF.
- Есть только теги чанков (`inpatient/ambulatory/any`), заполнены ~8%.
- `gather_protocol_text` для LLM их не фильтрует.
- В выдаче нет явного признака условий оказания.

---

## 3. План по трекам

Принцип: **Summary Cards и ICD-профили - источник истины**, свободный RAG - резерв;
каждый клинический пункт снабжается цитатой, страницей и вероятностью (support);
care_setting становится первоклассным полем и фильтром.

### Трек 1. Быстрые победы (низкий риск)

Задачи:
1. Включить дешёвый по памяти слой Summary-срезов: `PROTOCOL_SUMMARY_RAG_MERGE=1` (355 строк), при этом реранк оставить на precomputed-эмбеддингах, а не live-embed, чтобы не раздувать RAM. Задокументировать RAM-стоимость и оставить флаг переключаемым.
2. Sentence-aware excerpts: подключить `meaningful_excerpt.py` в `format_excerpt_for_display` / `gather_protocol_text` вместо жёсткой обрезки по символам. Резать по границам предложений; чинить дефисы переноса.
3. Отключить «отсебятину» для поиска по умолчанию: `RAG_EXTRACT_NON_PROTOCOL_FALLBACK=0` для protocol-search (в КЗ оставить). Пункты не из протокола - скрывать либо явно помечать.

Критерии приёмки:
- В excerpt-ах нет обрывков посреди слова/предложения (проверка на наборе запросов).
- В блоке «На приёме» нет строк `[не из протокола]` по умолчанию.
- Тесты на sentence-aware обрезку зелёные.

### Трек 2. Grounding препаратов и обследований с вероятностью (ядро запроса)

Задачи:
1. Для каждого извлечённого `medication` / `investigation` считать `support` в [0..1]:
   - fuzzy-совпадение строки с текстом чанков протокола;
   - косинус эмбеддинга пункта к чанкам протокола;
   - совпадение с `protocol_icd_profiles.jsonl` (обязательность `required/recommended`).
2. К каждому пункту прикреплять `source_ref` (страница + цитата) из чанка/summary.
3. Пункты ниже порога `support` - скрывать или помечать «требует проверки».
4. Приоритет структурированного слоя: список обследований/лечения строить сначала из `protocol_icd_profiles` + Summary Cards, LLM - только для формулировок поверх проверенных пунктов.

Критерии приёмки:
- Каждый показанный препарат/обследование имеет `support` и (по возможности) цитату+страницу.
- На контрольном наборе доля «неподтверждённых» пунктов в выдаче падает существенно.
- Есть тесты: пункт без опоры в тексте не проходит порог.

### Трек 3. care_setting как первоклассное поле

Задачи:
1. Добавить в каталог (`index.csv` / `protocol_catalog.jsonl`) поле `care_setting` на уровне PDF: агрегировать из имени файла (стационарных/амбулаторных условиях), тегов чанков и Summary `applicability`.
2. Скрипт пересборки каталога обновить (`scripts/build_protocol_catalog.py`).
3. В UI - бейдж «Стационар / Амбулаторно» на карточке протокола; фильтр в воронке.
4. В блоке «На приёме» - разделять препараты/обследования по условиям оказания, где это размечено.

Критерии приёмки:
- У каждого протокола в каталоге есть `care_setting` (или явное `unknown`).
- В выдаче виден признак условий оказания; фильтр работает.
- Тесты на парсинг care_setting из имени/тегов.

### Трек 4. Точность ранжирования и калибровка (современные подходы)

Задачи:
1. Cross-encoder reranker поверх top-N кандидатов (точнее bi-encoder cosine для top-1).
2. Заменить хардкод-штрафы `_rerank_protocols_symptom_only` на лёгкий обучаемый reranker по фичам (ICD-overlap, рубрика, audience, care_setting, семантика).
3. Калибровка уверенности протокола и пунктов на eval-наборе (`eval/`, `run_kz_vector_index_eval.py`), чтобы проценты отражали реальную точность (reliability curve).

Критерии приёмки:
- Метрики top-1 / top-3 на eval-наборе улучшаются относительно бейзлайна.
- Калиброванная уверенность: заявленный % близок к фактической точности.

### Трек 5 (фон). Полнота структурированного слоя
- Довести Summary Cards и `summary_chunks.jsonl` с 355 до всех 478 (`summary_to_rag.py`).
- Пройти ревью (`reviewed/` сейчас пуст).

---

## 4. Риски и ограничения

- RAM на Render: включение summary-merge и реранка увеличивает потребление. Митигировать precomputed-эмбеддингами и малым размером summary-слоя; держать флаги переключаемыми.
- Качество Summary Cards: часть в статусе `needs_review`/`auto_extracted`. Grounding-порог защищает от плохих карточек.
- Регрессии в КЗ: изменения общих функций (`gather_protocol_text`, excerpt) затрагивают и консультации - обязательны тесты обоих путей.

## 5. Порядок работ

Реализация итеративно: трек за треком, каждый - с тестами, коммитом, `git push` и подъёмом
`BUILD_VERSION`. После трека 1 - проверка на контрольных запросах, затем треки 2, 3, 4.

## 6. Затрагиваемые файлы (ориентир)

- `rag_server.py` (excerpts, gather_protocol_text, флаги, extract).
- `clinical_knowledge/meaningful_excerpt.py` (подключение в поиск).
- `clinical_knowledge/search_retrieval.py`, `protocol_icd_index.py` (структурный слой, allowlist).
- `clinical_knowledge/protocol_practical_lite.py` (grounding пунктов).
- `scripts/build_protocol_catalog.py`, `index.csv`, `data/protocol_catalog.jsonl` (care_setting).
- `index.html` (бейдж care_setting, показ support/цитат).
- `tests/` (новые тесты под каждый трек).
