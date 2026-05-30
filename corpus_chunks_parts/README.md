# Части `chunks.jsonl` (корпус corpus_pipeline)

Файл разбит скриптом `split_chunks_jsonl.py` на части **~45 МБ** (лимит GitHub на один файл — 100 МБ).

## Сборка обратно

```bash
cat corpus_chunks_parts/chunks.part.*.jsonl > output/chunks/chunks_merged.jsonl
```

После изменения корпуса пересоздайте части: `python3 split_chunks_jsonl.py` (нужен полный `output/chunks/chunks.jsonl`).
