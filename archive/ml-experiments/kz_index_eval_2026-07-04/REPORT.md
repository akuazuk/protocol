# KZ + Vector Index Eval

- **Дата:** 2026-07-04T15:13:45Z
- **Base:** https://protocol-bimy.onrender.com
- **BUILD:** 2026-07-04-r37-proto-semantic-lazy

## Vector index

| enabled | loaded | indexed | dim | path |
|---|---|---|---|---|
| True | False | 0 | 0 | `/var/data/corpus_vector_index` |

## Сводка

- Всего файлов: **41** (КЗ: 36, анализы A/a/А/а: 5)
- Успешно: **41**, ошибок: 0
- КЗ avg overall: **82.4%**
- КЗ overall<70%: **4**
- КЗ без RAG chunks: **36**
- КЗ semantic lexical (не vector): **0**

## План улучшений

1. P0: vector index не загружен на prod - проверить /var/data/corpus_vector_index и рестарт.
2. P1: 36 КЗ без RAG-чанков - усилить retrieval/allowlist: F_1_p, g_28_1, gastro_1, gyn_28_1, ja_1…
3. P1: 36 КЗ без retrieval_top - проверить vector prefilter и ICD routing.
4. P2: overall<70% у 4 КЗ - ручной разбор: ja_3, ja_4, report_n_1, report_n_2

## Weak KZ (overall < 70%)

- `ja_3` overall=55.0% rag=0 ret=0 sem=None
- `ja_4` overall=55.0% rag=0 ret=0 sem=None
- `report_n_1` overall=60.0% rag=0 ret=0 sem=None
- `report_n_2` overall=61.9% rag=0 ret=0 sem=None

## B2C анализы (A/a/А/а)

- `A_2` mismatch=True wrong=lab_in_kz overall=None
- `a_1` mismatch=True wrong=lab_in_kz overall=None
- `a_3` mismatch=True wrong=lab_in_kz overall=None
- `a_4` mismatch=True wrong=lab_in_kz overall=None
- `a_pl_1` mismatch=True wrong=lab_in_kz overall=None
