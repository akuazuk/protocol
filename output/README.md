# Структурированный корпус клинических протоколов (retrieval-ready)

Каталог создаётся скриптом `python3 -m corpus_pipeline.run_pipeline` из PDF в `minzdrav_protocols/` (или `CORPUS_PDF_ROOT`).

## Зависимости

```bash
pip install -r requirements-corpus-pipeline.txt
```

Опционально для OCR плохих PDF: `CORPUS_USE_OCR=1`, установить [Tesseract](https://github.com/tesseract-ocr/tesseract) и `pip install pytesseract Pillow`.

## Переменные окружения

| Переменная | Описание |
|------------|----------|
| `CORPUS_PDF_ROOT` | Каталог с PDF (по умолчанию `./minzdrav_protocols`) |
| `CORPUS_OUTPUT_ROOT` | Выход (по умолчанию `./output`) |
| `CORPUS_MIN_CHARS_PAGE` | Порог «мало текста» на странице (по умолчанию 80) |
| `CORPUS_USE_OCR` | `1` — при слабом текстовом слое вызывать OCR страницы |

## Выходные файлы

### `documents/*.json`

Один JSON на **логический документ** (один PDF может дать несколько, если внутри несколько блоков «УТВЕРЖДЕНО» / «КЛИНИЧЕСКИЙ ПРОТОКОЛ»).

Поля верхнего уровня:

| Поле | Описание |
|------|----------|
| `doc_id` | Уникальный id (хеш пути + опционально `L0`, `L1`…) |
| `pdf_doc_id` | Id исходного PDF без суффикса логической части |
| `source_path` | Путь от корня проекта |
| `file_name` | Имя файла |
| `title`, `subtitle` | Заголовок протокола (эвристика) |
| `act` | Реквизиты нормативного акта: `issuing_body`, `act_type`, `date`, `number`, `status`, `amendments[]`, `repeals[]` |
| `protocol_passport` | `protocol_title`, `clinical_domain`, `topic`, `population[]`, `care_setting[]`, `icd10_codes[]`, `key_terms[]`, … |
| `text.normalized` | Полный нормализованный текст логического документа |
| `text.pdf_raw_char_length` | Длина склеенного «сырого» текста PDF (для контроля) |
| `pages` | По страницам: `page_no`, `extraction_confidence`, `ocr_used`, `chars` |
| `chunk_count`, `table_count`, `page_count` | Статистика |
| `extraction_confidence` | Средняя уверенность по страницам |
| `page_offset_base` | Смещение логического фрагмента в полном тексте PDF |

### `chunks/chunks.jsonl`

По одной строке JSON на чанк (удобно для потоковой загрузки в OpenSearch / SQLite FTS).

| Поле | Описание |
|------|----------|
| `chunk_id` | Уникальный id |
| `doc_id` | Связь с документом |
| `section_id`, `section_path[]` | Раздел |
| `chunk_type` | Например `diagnostics`, `treatment`, `table_block`, `drug_list`, `criteria_block` |
| `text` | Текст чанка |
| `page_from`, `page_to` | Номера страниц (1-based) в исходном PDF |
| `point_numbers[]` | Обнаруженная нумерация пунктов |
| `icd10_codes[]`, `population`, `conditions`, `drugs`, `keywords`, `durations` | Извлечённые сущности (эвристика) |
| `embedding_ready_text` | Текст для эмбеддинга (раздел + МКБ + популяция + текст) |

### `tables/tables.json`

Массив таблиц: `table_id`, `pdf_doc_id`, `source_path`, `page`, `columns[]`, `rows[]`, `raw_markdown`, `normalized`, `extraction_confidence`.

### `entities/entities.json`

Агрегированные частоты: `icd10_codes`, `populations`, `care_settings`, `drugs`, `terms`, `procedures` (словари «строка → частота»).

### `registry/index.csv`

Реестр всех логических документов с колонками `doc_id`, `pdf_doc_id`, `source_path`, `file_name`, `title`, `logical_index`, `chunks`, `tables`, `pages`.

## Правила чанкинга

- Документ режется по **обнаруженным разделам** (диагностика, лечение, …) по ключевым словам в строках.
- Внутри раздела — по **нумерации** (`1.`, `1.1.`, `п. 12`) или по абзацам при длине > ~3500 символов.
- Таблицы извлекаются **отдельно** (pdfplumber), не дублируются в чанках построчно (строки таблицы можно добавить в следующей версии как `table_row` chunks).
- Фиксированный размер символов **не используется** как основной нож; при отсутствии структуры весь раздел может остаться одним чанком.

## Нормализация таблиц

- Колонки — первая строка таблицы.
- `raw_markdown` — табличное представление для отладки.
- Многостраничное объединение таблиц — заглушка; при необходимости дорабатывается по шапке колонок.

## Примеры поисковых запросов (после загрузки в Индекс)

- **МКБ-10:** поле `icd10_codes` в чанках / фильтр по `entities.icd10_codes`.
- **Название протокола:** `title` в `documents` или `registry`, полнотекст по `text.normalized`.
- **Нозология:** ключевые слова + `protocol_passport.key_terms` + `chunk.text`.
- **Лекарства:** `drugs` в чанке или ключ `entities.drugs`.
- **Диагностические критерии:** `chunk_type` + `conditions` или раздел `diagnostics`.
- **Пациентская группа:** `population` в чанке или `protocol_passport.population`.
- **Длительность:** поле `durations` в чанке (регулярные выражения по срокам).
- **Таблицы:** поиск по `tables.json` в `raw_markdown` / `rows`.

## Ограничения текущей версии

- Паспорт нормативного акта и разбиение на несколько протоколов в одном PDF — **эвристики** по тексту.
- Разделы и сущности — **регулярные выражения и словари**, без отдельной модели NER.
- Юридические цепочки «утратил силу» / «замена» — извлекаются только если фразы попали в первые тысячи символов шапки.
- `extraction_confidence` на странице снижается при коротком нативном тексте; при OCR — отдельно помечается `ocr_used`.

Для production рекомендуется валидация выборочных документов и при необходимости донастройка шаблонов в `corpus_pipeline/section_detect.py` и `passport_build.py`.
