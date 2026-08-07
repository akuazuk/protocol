# Контракт: mo_pipeline → gcp-llm → llm_inbox

GCP LLM **не** владеет warehouse. Только grades/judges.

## Input (из `$MO_DATA_ROOT/llm_outbox/<run_id>/` или GCS prefix)

| Файл | Смысл |
|--|--|
| `manifest.json` | run_id, day, model flags |
| `cases.jsonl` | минимальные слоты для grade (visit_id + clinical fields) |
| `llm_queue.json` | optional queue subset |

### manifest.json

```json
{
  "schema_version": 1,
  "run_id": "gcp-llm-2026-08-06-r1",
  "day": "2026-08-06",
  "escalate": true,
  "action_judge": true,
  "mo_action_judge_limit": 0
}
```

## Output (в `$MO_DATA_ROOT/llm_inbox/<run_id>/`)

| Файл | Смысл |
|--|--|
| `kz_l1_YYYY-MM-DD_llm_grades.jsonl` | night grades |
| `judges.jsonl` | action-judge rows (если включено) |
| `result_manifest.json` | counts, errors, model, cost_usd |

### result_manifest.json

```json
{
  "schema_version": 1,
  "run_id": "gcp-llm-2026-08-06-r1",
  "day": "2026-08-06",
  "grades_ok": 80,
  "grades_err": 0,
  "judges_ok": 155,
  "finished_at": "2026-08-07T14:00:00+00:00",
  "model_primary": "gemini-…"
}
```

## Запрещено

- запись в `warehouse/`, `gold_review/`, `crm_*`
- MIS password в env job (не нужен)
- полный corpus PDF

После validate BY/GCP app: atomic move grades → `secure_cases/…` + `recompute_mo_days`.
