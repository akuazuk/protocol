# Priority cases triage

Источник: `/Users/pavelkuzauka/Cursor_Folders/Protocol/ml/datasets/priority_cases.jsonl`
Кейсов (rating≤2): **22**

## Теги
- `score_misleading`: 20
- `false_positive_rule`: 11
- `wrong_protocol`: 4

## Кейсы

| hash | rating | verdict | rubric/tags |
|------|--------|---------|-------------|
| `sha256:ce6470d537b7f` | 2 | partially_wrong | false_positive_rule, wrong_protocol, score_misleading |
| `sha256:ce6470d537b7f` | 2 | partially_wrong | false_positive_rule, score_misleading |
| `sha256:ce6470d537b7f` | 2 | partially_wrong | score_misleading |
| `sha256:79709a76d406c` | 2 | partially_wrong | false_positive_rule, score_misleading |
| `sha256:3ac8ed06b5f48` | 1 | wrong | score_misleading |
| `sha256:ce6470d537b7f` | 2 | partially_wrong | score_misleading |
| `sha256:cac255f271432` | 1 | wrong | wrong_protocol |
| `sha256:3b06e56dcf784` | 2 | partially_wrong | false_positive_rule, score_misleading |
| `sha256:f6a9e0e8cdb2a` | 2 | partially_wrong | false_positive_rule, score_misleading |
| `sha256:25a31bd33b052` | 2 | partially_wrong | score_misleading |
| `sha256:d6c28f4581dab` | 2 | partially_wrong | false_positive_rule, score_misleading |
| `sha256:3cc7242ab500d` | 2 | partially_wrong | score_misleading |
| `sha256:b6149564b1755` | 2 | partially_wrong | false_positive_rule, score_misleading |
| `sha256:4888c4caac814` | 2 | wrong | wrong_protocol, false_positive_rule, score_misleading |
| `sha256:73304a4b3d721` | 2 | partially_wrong | score_misleading |
| `sha256:cac255f271432` | 2 | partially_wrong |  |
| `sha256:1f5795ecd54f8` | 2 | partially_wrong | false_positive_rule, score_misleading |
| `sha256:32f918828fa50` | 2 | partially_wrong | false_positive_rule, score_misleading |
| `sha256:1e68f0412e536` | 2 | partially_wrong | score_misleading |
| `sha256:cf199aaa009a9` | 2 | partially_wrong | score_misleading |
| `sha256:bfd725c9fd057` | 1 | wrong | wrong_protocol, false_positive_rule, score_misleading |
| `sha256:c442de90b248b` | 2 | partially_wrong | score_misleading |

## Рекомендуемые actions

- Engine: rule family gates / condition context (см. overrides в analysis_review)
- RAG: собрать retrieval_fix; проверить rubric pre-filter
- Compliance: caps sparse KZ, hybrid weights
