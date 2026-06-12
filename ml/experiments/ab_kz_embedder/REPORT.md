# A/B: baseline e5 vs fine-tuned embedder (KZ + RAG)

- **A:** `intfloat/multilingual-e5-small`
- **B:** `ml/experiments/embedder_exp_001/checkpoint_final`
- **Метод:** полный `retrieve()` на корпусе КП; тексты `consult_gold.jsonl` как запрос RAG

## Результаты

| Слой | A | B | Δ |
|------|---|---|---|
| Rule checker (9 КЗ) | 100% | 100% | 0 |
| RAG по тексту КЗ, топ-5 | 100% | 100% | 0 |
| Golden RAG (18 запр.) | 77,8% | 77,8% | 0 |

## Вывод

Анализ КЗ по правилам **не изменился** (embedder не участвует в send_gate).

На гастро-эталоне **полный RAG уже находил** нужные протоколы на обоих плечах.

Офлайн-прирост MRR (+0,29) из embedder_exp_001 **не проявился** на этих end-to-end метриках.

Следующий шаг: `retrieval_fix` от методиста, reranker, провалы golden (E11, K85, I50).

Пересчёт: `python3 scripts/run_ab_embedder_kz.py`
