# service: mis_bridge

**Образ:** `protocol-mis-bridge`  
**Эпохи:** E1 = Mac launchd, E2 = GCP Job, E3 = BY. Меняется только `RUN_HOST` / сеть.

## Владение

| Путь | Роль |
|--|--|
| `scripts/export_mis_protocol_month.py` и daily extract в mo daily | выгрузка МИС |
| `services/mis_bridge/extract_day.py` | thin CLI + upload staging |
| `deploy/mac-bridge/extract-contract.md` | контракт файлов |
| `requirements-mis-bridge.txt` | pymysql/sqlalchemy |

## Запреты

- Нет `google.generativeai` / Gemini.
- Нет `rag_server` / frontend.
- Пароль МИС только env (`KRAVIRA_DB_PASSWORD`); на GCE - `/opt/protocol/.env.mis`.
  Не в git и не в llm-образ.

## CLI

```bash
PYTHONPATH=. python3 -m services.mis_bridge.extract_day --help
# RUN_HOST=mac|gcp|by  OUT_DIR=.../inbound/extract
```
