# KZ + Vector Index Eval

- **Дата:** 2026-07-04T15:49:47Z
- **Base:** https://protocol-bimy.onrender.com
- **BUILD:** 2026-07-04-r37-proto-semantic-lazy
- Tier: **L2**

## Vector index

| enabled | loaded | indexed | dim | path |
|---|---|---|---|---|
| True | False | 0 | 0 | `/var/data/corpus_vector_index` |

## Сводка

- Всего файлов: **36** (КЗ: 7, анализы A/a/А/а: 0)
- Успешно: **7**, ошибок: 29
- КЗ avg overall: **79.1%**
- КЗ overall<70%: **1**
- КЗ без RAG chunks: **0**
- КЗ semantic lexical (не vector): **0**

## План улучшений

1. P0: vector index enabled но не в RAM - при первом запросе грузится ~1GB; на Render Standard возможен 502/OOM. Решение: mmap/lazy load или plan Pro.
2. P2: semantic probe failed для 7 КЗ - F_1_p, gastro_1, ja_1, ja_4
3. P2: overall<70% у 1 КЗ - ручной разбор: ja_4

## Weak KZ (overall < 70%)

- `ja_4` overall=55% rag=3 ret=3 sem=None

## B2C анализы (A/a/А/а)

