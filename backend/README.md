# Backend

Канонический backend-контур проекта.

## Точка входа

- `backend/server.py` -> `app` (re-export из `rag_server.py`)
- запуск:

```bash
uvicorn backend.server:app --host 127.0.0.1 --port 8787
```

## Что здесь должно жить дальше

- API и backend-сервисы (`api/`, `clinical_knowledge/`, `config/`);
- интеграции с МИС, методист, patient API;
- оркестрация ежедневных batch-пайплайнов.

Текущий этап - безопасная миграция структуры без изменения прод-логики.

