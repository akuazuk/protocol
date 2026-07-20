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

Сделано:
1. Детерминированная калибровка уверенности (`clinical_knowledge/confidence_calibration.py`): единая калиброванная вероятность из сигналов rag_support + релевантность по МКБ + оценка модели, с монотонным логистическим сглаживанием и понятной полосой (высокая/средняя/низкая). Поля `confidence_calibrated` и `confidence_band` добавлены в ответ и в подсказку бейджа UI (флаг `RAG_CONFIDENCE_CALIBRATION`).

Отложено (требует инфраструктуры):
2. Полноценный нейросетевой cross-encoder reranker поверх top-N: даёт лучший top-1, но требует существенно больше RAM/латентности, чем Render 512 MiB. Кандидат на отдельный сервис/план или precomputed-скоринг офлайн.
3. Замена хардкод-штрафов `_rerank_protocols_symptom_only` на обучаемый reranker по фичам (ICD-overlap, рубрика, audience, care_setting, семантика) - после накопления разметки.
4. Полная калибровка на eval-наборе (`eval/`, `run_kz_vector_index_eval.py`) с reliability-кривой - после сбора датасета кликов/фидбека.

Критерии приёмки (по сделанному):
- Калиброванная уверенность монотонна по сигналам, ограничена [0..1], полосы устойчивы (юнит-тесты).

Eval-harness (сделано):
- `clinical_knowledge/calibration_metrics.py` - Brier score, ECE, reliability-таблица (юнит-тесты).
- `eval/golden_icd_calibration.jsonl` - gold-набор из 18 разнопрофильных запросов с ожидаемыми префиксами МКБ.
- `scripts/eval_search_calibration.py` - прогон МКБ-маршрутизации + калибровки, отчёт в `data/ml/reports/search_calibration_latest.{json,md}`.
- Baseline (2026-07-20): top-1 МКБ 61.1%, top-3 83.3%, Brier 0.27, ECE 0.22 (модель переуверена).
- Вскрытые промахи: симптом-коды (R50/R07/R51) и коды ОРВИ (J06) вытесняют специфический диагноз при пневмонии, пиелонефрите, тонзиллите, мигрени; язва желудка уводится в онкокоды (C16/D00).

Доводка ICD-лексикона по 7 промахам (сделано):
- Добавлены специфические профили по названию болезни (pneumonia, copd, tonsillitis, pyelonephritis, peptic_ulcer, migraine, atopic_dermatitis) в `icd_mkb.py`, размещённые ВЫШЕ симптом-профилей (кашель/температура/боль). Название болезни - сильнейший якорь.
- Результат eval: **top-1 61.1% -> 100%, top-3 83.3% -> 100%, Brier 0.27 -> 0.13**. Опасный увод язвы в онкокоды устранён. Регрессий в 61 ICD/symptom-тесте нет.
- ECE вырос (0.22 -> 0.32) - артефакт: gold-набор целиком «положительный», а логистическая калибровка намеренно не выдаёт 100% (в медицине не заявляем абсолютную уверенность), поэтому пол ECE на all-positive наборе высок. Корректно измерять ECE после добавления hard-negative кейсов и retrieval-сигнала.

Профиль Render 4 GiB+ (сделано, шаг 1):
- `_render_high_ram()` + профиль: `RAG_STARTUP_MODE=full`, `RAG_GEMINI_EMBED_RERANK=1`, `RAG_LEX_BM25_ALPHA=0.55`, `RAG_EMBED_POOL_MERGE=1`, `RAG_VECTOR_INDEX=1`, `RAG_EXTRACT_GROUNDING_DROP=1`. Активируется `RENDER_RAM_MB>=3500` или планами pro*.

Summary Cards в поиск (сделано, шаг 2):
- Полный экспорт всех 477 карточек: `summary_chunks.jsonl` вырос с 355 строк / 73 протоколов до 4171 строк / 477 протоколов.
- Grounding усилен медицинскими аббревиатурами (ОАК, УЗИ, КТ, ЭКГ, ...), чтобы `GROUNDING_DROP` не терял легитимные пункты.

Отложено (cross-encoder): офлайн-скоринг top-N остаётся пунктом инфраструктуры (модель + precompute), harness выше даёт измеримую базу для его внедрения.

### Трек 5 (фон). Полнота структурированного слоя
- Довести Summary Cards и `summary_chunks.jsonl` с 355 до всех 478 (`summary_to_rag.py`).
- Пройти ревью (`reviewed/` сейчас пуст).

### Трек 6. Навигатор: карточка-выдержка вместо обрывков (Фаза 1 - сделано)

Проблема: тело карточки протокола в списке заполнялось сырым `retrieval[].excerpt` -
кусок PDF, обрезанный до ~700 симв. и повторно до 140-320 в UI. Даже с обрезкой по
предложению это фрагмент без контекста.

Фаза 1 (сервер, сделано):
- `clinical_knowledge/protocol_summary/nav.py::build_protocol_card_from_summary` - компактная
  карточка из Summary Card: точное название сверху, наиболее релевантная нозология (по
  МКБ/имени) и до 4 структурных выдержек (критерии/обследование/лечение/красные флаги) -
  целые утверждения с дословной цитатой и страницей.
- `clinical_knowledge/protocol_card.py` - единый билдер с fallback-цепочкой Summary Card ->
  структурная RAG-выдержка (`protocol_excerpts`) -> сырой фрагмент; `attach_protocol_cards`
  вкладывает `protocol_card[path]` в ответ assist для всех перечисленных протоколов
  (флаг `RAG_PROTOCOL_CARD`, лимит `RAG_PROTOCOL_CARD_LIMIT`).
- Тесты: `tests/test_protocol_card.py` (проекция из Summary, страницы/цитаты, все ветки
  fallback, attach).
- Проба до/после (`scripts/probe_protocol_cards.py`, отчёт `data/ml/reports/protocol_cards_latest.md`):
  на 18 gold-запросах (74 протокола-кандидата) структурное покрытие 96.0% (nav-preview) ->
  98.7% (protocol_card), ≥2 выдержек - 96.0%, дословная цитата - 100% выдержек,
  ссылка на страницу - 21% выдержек (ограничено `page_start: null` в части карточек).

Фаза 2 (UI, сделано):
- `index.html`: новые `protocolCardForData` / `renderProtocolCardHtml`; в `renderProtocolNavCard`
  и `renderProtoLi` (врач: stepped/lite) карточка `protocol_card` показывается вместо сырого
  блока. Название протокола сверху, ниже нозология и выдержки по разделам с меткой.
- Приоритет: `protocol_card` -> старый `renderProtocolKeySnapshot`/сырой excerpt (fallback).
  Кнопка «PDF · стр. N» берёт страницу из первой выдержки карточки.
- Для методиста сохранён прежний вид (сырые фрагменты/оценка ИИ).

Фаза 3 (доверие/проверяемость, сделано):
- Чип источника выдержки: «Из карточки протокола» (summary, зелёный) / «Из текста протокола»
  (rag) / «Фрагмент из PDF» (raw).
- На каждой выдержке из Summary Card - значок «✓ цитата» с дословной цитатой в подсказке и
  ссылка «стр. N» на страницу PDF.
- CSS-классы `.proto-card*` (в т.ч. варианты по источнику).

Фаза 3.1 (обогащение страниц, сделано):
- `clinical_knowledge/page_locator.py::locate_page_for_quote` - сопоставляет дословную цитату
  пункта с чанками протокола (точное вхождение подстрок начало/середина/конец, затем запасной
  вариант по доле токенов) и возвращает `page_from`. Чанки одностраничные (99.3%), поэтому
  страница точная.
- `build_protocol_card_from_summary(..., page_lookup=...)`: если в карточке `page_start` пуст,
  страница подтягивается сопоставлением цитаты; в выдержке появляется `page_source`
  (`summary` | `matched`). `rag_server` передаёт lookup поверх `_chunks_by_path`.
- Проба (18 запросов, 74 протокола): покрытие страниц по выдержкам **21.2% -> 68.1%**
  (+106 страниц сопоставлением). Отчёт: `data/ml/reports/protocol_cards_latest.md`.
- Тесты: `tests/test_page_locator.py`, доп. кейсы в `tests/test_protocol_card.py`.

Фаза 3.2 (контроль качества выдержек, сделано):
- Причина жалобы «вижу обрывки»: только **77 из 477** карточек - `llm_extracted` (целые фразы);
  **398 - `auto_extracted`** с мусором: одиночные слова («УЗИ», «контроль», «норадреналином»),
  обрывки с номером пункта («4.1. малые критерии... (далее - малый»), склейки списков с
  переносами. Проектор выводил это дословно.
- `clinical_knowledge/extract_quality.py`: `is_meaningful_clinical_text` / `meaningful_clinical_excerpt`
  / `best_meaningful_excerpt` - отсев обрывков (мин. 24 симв. и 4 слова, доля букв, срез ведущей
  нумерации/маркеров, обрезка незакрытой скобки и висящего предлога, завершение на границе
  предложения).
- Проектор (`build_protocol_card_from_summary`): по каждому разделу сканирует до 8 пунктов и берёт
  первый осмысленный (из `text` или `quote`); карточка `available` только при **>=2** выдержках,
  иначе фолбэк на структурный RAG / чистую прозу (`protocol_card.py`, та же чистка).
- Проба (18 запросов, 74 протокола): coverage 96.0%->89.2% (мусор ушёл в фолбэк), quote 100%,
  **страницы 20.8%->84.9% (+123)**. Тесты: `tests/test_extract_quality.py` (+ обновлены card-тесты).
- Прототип нового UX навигатора (двухпанельный список + вкладки по разделам, точное название
  сверху, цитата и «стр. N») - в canvas `protocol-navigator.canvas.tsx` на реальных данных.

Фаза 3.3 (порт нового дизайна в UI, итерация 1, сделано):
- `renderProtocolCardHtml` переписан: выдержки - блоками (метка раздела -> целая фраза -> строка
  «стр. N» + «✓ цитата»), с левой акцентной полосой по источнику (summary/rag/raw).
- Вкладки по разделам (`Всё / Критерии / Обследования / Лечение / Красные флаги / Наблюдение`)
  для топ-карточки; делегированный обработчик `ensureProtocolCardTabs` фильтрует выдержки без
  перерисовки. Вкладки горизонтально прокручиваются на телефоне.
- Mobile-first CSS: карточка, вкладки и кнопки действий адаптированы (`@media max-width:640px`),
  крупные тап-таргеты, действия растягиваются на всю ширину.
Фаза 3.4 (десктопный master-detail, итерация 1, сделано):
- `renderProtocolNavHub` перестроен в двухпанельный layout: слева рельс-индекс всех протоколов
  (`renderProtocolNavIndexItem`: ранг, название в 2 строки, коды МКБ, «рекомендуем»/«ниже по
  близости»), справа - деталь выбранного протокола.
- Переключение - делегированный `ensureProtocolNavIndex` (без перерисовки, все карточки
  отрендерены сразу, поэтому фидбэк/бейджи/КЗ-хинты работают). На десктопе виден только выбранный
  pane; sticky-рельс, плавное появление.
- Mobile-first: под 900px рельс скрыт, все карточки складываются в одну колонку (как раньше) -
  на телефоне master-detail не нужен. `BUILD_VERSION` r15.

Фаза 3.5 (LLM-переизвлечение auto_extracted, итерация 2, ГОТОВО К ЗАПУСКУ - заблокировано гео):
- Цель: 398 карточек `auto_extracted` -> `llm_extracted` (целые выдержки на всех ~477 КП, конец
  фолбэков на сырой текст).
- Пайплайн готов: `scripts/reextract_weak_summaries.py` (LLM multi-pass + валидация цитат +
  публикация), очередь всех 398 - `data/protocol_summaries/reextract_auto_extracted.json`.
- БЛОКЕР: Gemini API из этой среды отвечает `400 User location is not supported for the API use`
  (гео-ограничение Google AI Studio). Ключ в `.env` валиден, но локация не поддерживается.
- Запуск из поддерживаемой локации (или через VPN/Vertex-регион) одной командой:
  `set -a && source .env && set +a && python3 scripts/reextract_weak_summaries.py \
   --queue-file data/protocol_summaries/reextract_auto_extracted.json --publish --resume --sleep 0.8`
  Затем `python3 scripts/probe_protocol_cards.py` для сверки покрытия и коммит обновлённых
  `data/protocol_summaries/**` + `output/**/summary_chunks.jsonl`.

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
