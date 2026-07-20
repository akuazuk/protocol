# Protocol Summary Schema (v1.0)

Источник истины: `clinical_knowledge/protocol_summary/schema.py`.

- **ProtocolSummary** - карточка одного PDF/протокола
- **ConditionSummary** - нозология внутри протокола (МКБ, exams, treatment, red flags)
- **SummarySourceRef** - обязательная привязка к странице/разделу/цитате
- Статусы: `extraction_status`, `review_status`, `validation.status`

Хранение: `data/protocol_summaries/yaml/{protocol_id}.yaml`
