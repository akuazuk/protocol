# service: api

**Образ:** часть `protocol-gcp-app` (и будущий `protocol-by-home`).  
**Не включает:** Gemini night grade, MIS SQL extract.

## Владение (канон сейчас)

| Путь | Роль |
|--|--|
| `rag_server.py` | FastAPI entry |
| `frontend/web/` | статика doctor/methodist/patient |
| `backend/` | server helpers |
| `requirements-rag.txt` | deps образа api |

## Запреты

- Не импортировать MIS DSN / sql_epam в hot path web.
- Не запускать `grade_kz_llm` / action-judge из request handlers.
- Не `COPY` в образ: `minzdrav_protocols/**/*.pdf`, `data/medical_exams`, gold dumps.
