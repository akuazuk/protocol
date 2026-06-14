# План улучшения поиска протоколов и навигации (v2)

**Проект:** Protocol  
**Версия:** 2.0  
**Дата:** июнь 2026  
**Сборка:** r133+  
**Цель:** врач быстро находит нужный PDF и раздел; методист оценивает качество RAG через ИИ и подтверждает правки для движка.

---

## 1. Роли: кто что делает

| Роль | Задача | Инструмент |
|------|--------|------------|
| **Врач** | Диагноз/МКБ → релевантный протокол → цитата/PDF | Вкладка «Поиск протоколов», lite UI |
| **ИИ (методист)** | Мета-оценка ranking: verdict, теги, retrieval_fix, правки RAG | `POST /api/methodist/search-ai-review` |
| **Методист** | Одобрить / отклонить / поправить → feedback | Панель под результатами поиска |
| **Движок** | BM25, embed rerank, ICD, dedup, golden eval | `rag_server.py`, `methodist_stats.search` |

**Принцип (как для КЗ):** ИИ предлагает, человек отвечает за финальную метку.

---

## 2. Что оценивать в режиме методиста (поиск)

### 2.1 Обязательно (каждый прогон с ошибкой или сомнением)

1. **Top-1 релевантен?** (`top1_relevant`, `ranking_rating` 1–5)
2. **Неверный протокол в top** → `wrong_protocol` + `rejected_path`
3. **Нужный КП не в top-3** → `missed_protocol` + `chosen_path` (autocomplete)
4. **Детский/взрослый, беременность** → `wrong_population`
5. **Только жалобы без МКБ** → `query_too_vague` (подсказка врачу: «Подобрать коды МКБ»)

### 2.2 Сохраняется в feedback

| Событие | Когда |
|---------|--------|
| `search_review` | ИИ: выдача верна, методист одобрил |
| `retrieval_fix` | Есть rejected/chosen; + `retrieval_top_paths` для Hit@k |
| `protocol_search` | Телеметрия каждого `/api/assist` (без текста запроса) |

### 2.3 Не оценивает методист на поиске

- Клиническое содержание КZ (это вкладка «Проверка КЗ»)
- Юридические/МЭЭ-вердикты
- Замена очного приёма

---

## 3. Поток для врача (навигация)

```text
Жалобы / диагноз / МКБ
  → [опционально] «Подобрать коды МКБ»
  → «Найти протоколы»
  → Список PDF (dedup), % , top-1 match_reason
  → «Открыть PDF» | «Разобрать протокол»
       → выдержка + TOC Summary (если есть YAML)
       → матрица КЗ по кнопке (не автоматически)
```

### 3.1 Уже сделано (r128–r132)

- Dedup PDF по basename
- Lite assist: без summary/ICD/diff на главной
- «Разобрать протокол» только по кнопке
- Режим «только цитаты»
- Methodist UI без маркетинговых блоков

### 3.2 Следующие шаги для врача (P2–P4)

| ID | Улучшение | Эффект |
|----|-----------|--------|
| N1 | **Маршрут «жалобы → МКБ → протокол»** — явный stepper в UI | Меньше пустых поисков |
| N2 | **Summary-first retrieval** при явном МКБ | Hit@1 ↑ |
| N3 | **TOC внутри «Разобрать»** — якоря: диагностика, лечение, госпитализация | Навигация без чтения всего PDF |
| N4 | **Единый экран «Оформить КЗ по протоколу»** — протокол + чек-лист блоков КZ | Связка поиск ↔ КZ |
| N5 | **Кэш protocol-practical** по path+ICD | Быстрее повторные открытия |
| N6 | **Два этапа assist** — сначала retrieval (<15 с), потом опционально LLM | Скорость на слабом канале |

---

## 4. Поток для методиста (r133)

```text
Поиск в ?mode=methodist
  → POST /api/assist (compact)
  → POST /api/methodist/search-ai-review
  → Панель: rating, verdict, улучшения RAG
  → Методист: [Одобрить] | [Править] | [Отклонить ИИ]
  → POST /api/ml/feedback (search_review | retrieval_fix)
  → Дашборд «Поиск · оценки»
```

---

## 5. Дашборд «Поиск · оценки» (r133)

Вкладка методиста `#search-dashboard`:

| KPI | Описание |
|-----|----------|
| Поисков (telemetry) | `protocol_search` events |
| retrieval_fix (UI) | `source=protocol_search_ui` |
| search_review | Одобренная верная выдача |
| Hit@1 / Hit@3 | По разметкам с `retrieval_top_paths` |
| AI одобрено / assisted | Замыкание контура ИИ→методист |
| Готовность | Пороги 20 / 30 / 15 |

Таблицы: **приоритетные улучшения RAG** (из `engine_improvements_ru`), **последние разметки** (без текста запроса).

---

## 6. Дорожная карта по фазам

### Фаза A — Контур методиста для поиска ✅ r133

- [x] AI meta-review поиска
- [x] Одобрить / править / отклонить
- [x] `search` block в `/api/methodist/stats`
- [x] Дашборд «Поиск · оценки»
- [x] `retrieval_top_paths` на save

### Фаза B — Качество retrieval (2–4 нед.)

| ID | Задача |
|----|--------|
| B1 | Summary-first при МКБ в query |
| B2 | Pre-filter рубрика + МКБ до embed rerank |
| B3 | Golden queries (symptom / МКБ / mixed) + CI eval |
| B4 | `GET /api/methodist/queue?domain=search` — очередь плохих AI-verdict |
| B5 | Обязательный `rejected_path` при `wrong_protocol` |

### Фаза C — Навигация врача (3–5 нед.)

| ID | Задача |
|----|--------|
| C1 | Stepper жалобы→МКБ→протокол |
| C2 | Lazy KZ matrix (только запрошенный блок) |
| C3 | Protocol Summary TOC — клик → excerpt API |
| C4 | Экран «КЗ по выбранному протоколу» |

### Фаза D — ML (после ≥20 search retrieval_fix)

- LoRA embedder на `retrieval_pairs_resolved.jsonl`
- A/B summary-first vs legacy
- Эффект deploy на golden set (S6 из roadmap v1.1)

---

## 7. Метрики успеха

| Метрика | Сейчас | Цель B |
|---------|--------|--------|
| `retrieval_fix` (search UI) | <10 | ≥20 |
| Hit@3 (search labels) | низкий | ≥60% |
| AI-approved / assisted | 0 | ≥70% согласия |
| Доля поисков с МКБ | — | +15% |
| Дубликаты PDF в top-5 | 0 | 0 |

---

## 8. API (r133)

| Method | Path | Описание |
|--------|------|----------|
| POST | `/api/methodist/search-ai-review` | ИИ-оценка выдачи |
| GET | `/api/methodist/stats` | + поле `search` |
| POST | `/api/ml/feedback` | `search_review`, `retrieval_fix` |

---

## 9. Связанные документы

- [search-funnel-v1.md](./search-funnel-v1.md) — воронка 0–7, issues фаз B/C  
- `docs/search-methodist-roadmap.md` — v1.1 baseline  
- `docs/methodist-workbench-tz.md` — КЗ workbench  
- `clinical_knowledge/methodist_search_ai_review.py` — промпт поиска

---

## 10. История

| BUILD | Изменения |
|-------|-----------|
| r133 | Search AI review, search_review event, дашборд «Поиск · оценки», plan v2 |
| r132 | Methodist UI cleanup |
| r131 | Lite assist + doctor UI |
