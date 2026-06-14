# Roadmap: поиск протоколов и кабинет методиста

**Проект:** Protocol  
**Версия:** 1.0  
**Дата:** июнь 2026  
**Цель:** точный подбор КП Минздрава РБ по симптомам / диагнозу / МКБ-10 и навигация по протоколу под запрос; замкнутый контур улучшения через методиста (как для КЗ).

---

## 1. Текущая архитектура (as-is)

```text
Запрос + МКБ + рубрики
  → POST /api/assist
  → ICD pipeline → retrieve (BM25 + routing + embed rerank)
  → Gemini JSON (protocols, summary, differential)
  → confidence_score = blend(LLM, rag_support)
  → UI: список PDF + excerpt + optional clinical_detail
```

| Слой | Файлы | Роль |
|------|-------|------|
| UI | `index.html` — `#panel-main-search`, `runAssist`, `renderLlmOut` | Ввод, результаты, PDF |
| API | `rag_server.py` — `/api/assist`, `/api/protocol-detail`, `/api/icd-suggest` | Поиск и выдержки |
| RAG | `retrieve()`, `corpus_chunks_parts/`, `summary_chunks.jsonl` | Отбор фрагментов |
| Summary | `clinical_knowledge/protocol_summary/` | YAML-карточки (частичное покрытие) |
| Feedback | `search_telemetry.py`, `feedback_store.py` | Телеметрия поиска; `retrieval_fix` только из КЗ |
| Methodist | `methodist-workbench-tz.md`, вкладка КЗ | Разметка КЗ, retrieval_fix при wrong/missed protocol |

**Главный gap:** нет разметки выдачи на вкладке поиска; навигация — «PDF + один extract», а не оглавление summary под запрос.

---

## 2. Целевая архитектура (to-be)

```text
Query → condition match (Summary) → ranked protocols → section TOC → PDF evidence
                ↑
        Methodist search workbench → retrieval_fix → embedder + routing
```

PDF остаётся юридическим источником; **Protocol Summary** — машиночитаемый слой для поиска и навигации; **методист** — контур качества.

---

## 3. Фазы работ

### P0 — продукт без ML (текущий спринт)

| ID | Задача | Статус |
|----|--------|--------|
| P0.1 | Вкладка «Поиск» доступна в режиме методиста; панель `retrieval_fix` после `/api/assist` | ✅ r127 |
| P0.2 | TOC из Protocol Summary под каждым протоколом в выдаче; `GET /api/protocol-summary-nav` | ✅ r127 |
| P0.3 | `GET /api/methodist/protocol-search` — autocomplete для «Правильный протокол» | ✅ r127 |
| P0.4 | Query-aware вкладки focus в UI (investigations / medications / …) | частично (клик по секции TOC) |

### P1 — качество retrieval (2–4 недели)

- Summary-first retrieval для запросов с МКБ (`condition` match → PDF evidence).
- Инвертированный индекс МКБ + рубрика (pre-filter до embed rerank).
- Очередь «плохих поисков» в `GET /api/methodist/queue?domain=search`.
- Расширение golden set с поисковой вкладки.

### P2 — корпус и ML (после ≥20 retrieval_fix)

- Массовая генерация/валидация Protocol Summary для каталога.
- LoRA embedder на `retrieval_pairs_resolved.jsonl` (`docs/ml-backlog-when-kz-ready.md`).
- A/B: summary-first vs legacy на eval + methodist hit@3.

### P3 — умная навигация

- Вертикальный маршрут: жалобы → МКБ → протокол → раздел → цитата PDF.
- Сценарии методиста для ординаторов (эталон path + section).

---

## 4. API (новые и расширенные)

### `GET /api/protocol-summary-nav`

Публичный. Параметры: `path` (обяз.), `query`, `icd` (опционально).

Ответ: `{ available, protocol_id, title, conditions[{ name, icd10_codes, sections[{ id, label, count, extract_focus, preview }] }] }`

### `GET /api/methodist/protocol-search?q=…`

Только `X-Methodist-Token`. Top-10 по `index.csv` + title из Summary YAML.

### `POST /api/ml/feedback` — `retrieval_fix` с поиска

```json
{
  "event_type": "retrieval_fix",
  "query": "текст запроса assist",
  "rejected_path": "minzdrav_protocols/…",
  "chosen_path": "minzdrav_protocols/…",
  "note": "опционально",
  "source": "protocol_search_ui",
  "reviewer": "инициалы"
}
```

---

## 5. UI: панель методиста на поиске

Показывается при `body.methodist-mode` после успешного `runAssist`:

1. Read-only: top-5 retrieval paths + финальный ranking.
2. Select «Система ошибочно выбрала» (из top-5).
3. Autocomplete «Правильный протокол».
4. Теги: `wrong_protocol`, `missed_protocol`, `wrong_population`, `query_too_vague`.
5. Кнопка «Сохранить retrieval_fix».

---

## 6. UI: TOC из Summary

Под каждым протоколом в списке (если есть YAML):

- `<details>` «Навигация по карточке протокола».
- Нозологии (conditions) → секции с счётчиками пунктов.
- Клик по секции → `POST /api/protocol-detail` с `extract_focus`.

Fallback: только PDF + excerpt (как сейчас).

---

## 7. Метрики успеха

| Метрика | Сейчас | Цель P1 |
|---------|--------|---------|
| `retrieval_fix` (feedback) | ~9 | ≥20 |
| Hit@3 на размеченных парах | низкий | ≥60% |
| Доля поисков с МКБ (telemetry) | — | рост после UX |
| Summary coverage (рубрики) | частично | топ-8 рубрик ≥80% PDF |

---

## 8. Связанные документы

- `docs/methodist-workbench-tz.md` — §4.2.3 RAG, §5.5 protocol-search
- `docs/protocol_summary_schema.md` — схема карточек
- `docs/ml-backlog-when-kz-ready.md` — пороги ML
- `docs/methodist-ml-priority-plan.md` — опора A (подбор протокола)

---

## 9. История версий

| Версия | BUILD | Изменения |
|--------|-------|-----------|
| r127 | `r127-search-methodist-p0` | P0.1–P0.3: roadmap, methodist panel, summary nav API |
