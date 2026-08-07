# service: mo_pipeline

**Образ:** часть `protocol-gcp-app` (без Gemini SDK в идеале; transitional может делить образ с api).  
**Host:** GCP в E1/E2, BY в E3.

## Владение

| Путь | Роль |
|--|--|
| `clinical_knowledge/mo_*.py`, `kz_deep_eval.py`, `kz_evaluation_*.py` | score / warehouse |
| `scripts/recompute_mo_days.py` | отчёты из артефактов |
| `scripts/rescore_mo_deep_days.py` | deep rescore + CSV join |
| `scripts/backfill_mo_warehouse.py` | warehouse |
| `$MO_DATA_ROOT/warehouse`, `secure_cases`, `reports`, `inbound/extract` | данные |

## Запреты

- Не ходить в MariaDB напрямую (это `mis_bridge`).
- Не звать Gemini (это `llm_worker`); читать только `llm_inbox/`.
- Leader lock: один writer warehouse на эпоху.
