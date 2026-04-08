# Тесты (pytest)

## Зависимости

Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-rag.txt -r requirements-dev.txt
```

## Запуск

```bash
cd /path/to/Protocol
source .venv/bin/activate
python3 -m pytest tests/ -v
```

Полный прогон (pytest + golden на мини-корпусе):

```bash
./eval/run_all.sh
```

Переменные для изолированного корпуса задаёт `tests/conftest.py` (`tests/fixtures/chunks.mini.jsonl`, без Gemini embed rerank). Дополнительные переменные окружения обычно не нужны.

## Содержимое

| Файл | Назначение |
|------|------------|
| `conftest.py` | Мини-корпус и флаги до импорта `rag_server` |
| `fixtures/chunks.mini.jsonl` | Минимальный JSONL для смоук-retrieve |
| `test_health.py` | `/health`, `/api/specialties` |
| `test_retrieve_smoke.py` | Базовый смоук `retrieve()` |
| `test_retrieve_detailed.py` | Схема ответа, два PDF в фикстуре, негативный запрос, МКБ |
| `test_retrieval_checks.py` | Юнит-тесты `eval/retrieval_checks.py` (без `rag_server`) |

## Ручная проверка запросов (eval)

Интерактивный просмотр результатов `retrieve()` и прогон golden-файла:

```bash
python3 eval/query_tester.py --mini --query "ваш запрос"
python3 eval/query_tester.py --mini --golden eval/golden_queries.jsonl
```

Без `--mini` нужен полный корпус в `RAG_CHUNKS_*` (как у production).

Для **оценки качества с диагностикой и планом** (полный `retrieve`, опционально embed-rerank и совет Gemini) см. `eval/README.md` и скрипт `eval/search_quality_eval.py`.
