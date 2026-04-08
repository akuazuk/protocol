# Оценка качества поиска (golden queries)

Файл `golden_queries.jsonl` — по одному JSON на строку:

- `query` — текст запроса к RAG (как в UI или `retrieve()`).
- `must_substrings` — подстроки в объединённом тексте топ-чанков (регистр не важен).
- `expected_any_path_contains` — фрагменты пути к PDF; успех, если в топе есть путь с любым из них.
- `forbidden_substrings` — не должны встречаться в тексте топа (анти-паттерны).
- `min_chunks` — минимальное число строк выдачи.
- `expect_empty` — `true`: успех только при пустом отборе (негативные запросы).
- `notes` — комментарий.

Общая логика проверок вынесена в `eval/retrieval_checks.py` (используется и в pytest).

## Быстрый смоук: `query_tester.py`

Лексическая проверка `retrieve()` (можно без embed-rerank):

```bash
python3 eval/query_tester.py --mini --golden eval/golden_queries.jsonl
```

## Точная оценка поиска: `search_quality_eval.py`

Полный `retrieve()` как в приложении, включая **Gemini embed-rerank**, если заданы `RAG_GEMINI_EMBED_RERANK=1` и `GOOGLE_API_KEY`. Для каждого кейса выводит диагностику, **эвристический план** исправлений и при флаге **`--gemini-advice`** — краткий анализ и шаги от модели.

```bash
# мини-корпус, без ключа (rerank отключится сам, в отчёте будет подсказка включить ключ)
python3 eval/search_quality_eval.py --embed-off --mini --golden eval/golden_queries.jsonl

# как ближе к проду: семантический rerank + при необходимости совет модели
export GOOGLE_API_KEY=...
python3 eval/search_quality_eval.py --embed-on --golden eval/golden_queries.jsonl --gemini-advice --report-json eval/last_report.json
```

Скрипт печатает **диагностику**, **план (эвристика)** и блок **«Доп. подсказки»** (разрыв скоров, один доминирующий PDF, низкий top_score). При провалах golden выводится **сводка по критериям** (`must_substrings`, `expected_path`, …). JSON-отчёт: `--report-json путь.json`.

Шаблон эталонов для **полного** корпуса: `eval/golden_queries.prod.example.jsonl` — скопируйте в `golden_queries.prod.jsonl` (удобно добавить в `.gitignore` локально).

Одна команда из корня репозитория: `./eval/run_all.sh` (pytest + этот скрипт на мини-корпусе).

Без `--mini` используется полный корпус из `RAG_CHUNKS_*` (как у сервера).
