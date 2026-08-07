# service: llm_worker

**Образ:** `protocol-gcp-llm`  
**Host:** только GCP (не Mac geo). Канон данных - не здесь.

## Владение

| Путь | Роль |
|--|--|
| `scripts/grade_kz_llm.py` | night grades |
| `scripts/run_mo_action_queue_llm_judge.py` | action-judge |
| `services/llm_worker/grade_day.py` | thin CLI под job-contract |
| `deploy/gcp-llm/job-contract.md` | input/output |
| `requirements-llm-worker.txt` | минимальный Gemini stack |

## Запреты

- Нет MIS DSN / sql_epam.
- Нет записи в `warehouse/` или `gold_review/` - только `llm_inbox/` (или stdout paths из контракта).
- Не тащить PDF corpus и pandas pipeline.

## CLI

```bash
PYTHONPATH=. python3 -m services.llm_worker.grade_day --help
```
