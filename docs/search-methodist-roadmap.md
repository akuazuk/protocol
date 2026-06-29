# Roadmap: поиск протоколов и кабинет методиста

**Проект:** Protocol  
**Версия:** 1.1  
**Дата:** июнь 2026  
**Цель:** точный подбор КП Минздрава РБ по симптомам / диагнозу / МКБ-10; компактная выдача для врача и методиста; замкнутый контур улучшения RAG через `retrieval_fix` и статистику (как для КЗ).

---

## 1. Два режима вкладки «Поиск протоколов»

### 1.1 Обычный режим (врач / ординатор)

**Задача:** быстро найти релевантный PDF и понять, насколько он подходит к запросу.

| Элемент | Поведение |
|---------|-----------|
| Ввод | жалоба / диагноз / МКБ; опционально «Подобрать коды МКБ» |
| Результат | уникальные PDF (dedup), % соответствия, **оценка ИИ** (`match_reason`), цитата из PDF, ссылка |
| Краткий ответ модели | блок `summary` — опционально; скрыт в «только цитаты» |
| Разбор / КЗ | **только по кнопке** «Разобрать протокол» — выдержка + матрица КЗ |
| Дифференциал, уточняющие вопросы | показываются; не блокируют основной список |
| TOC Summary | под протоколом (если есть YAML); клик → выдержка по разделу |

**Не показывать по умолчанию:** автозагрузку практического разбора, матрицу КЗ на десятки пунктов, дубликаты одного PDF из разных рубрик.

### 1.2 Режим «Только цитаты» (аудит)

Чекбокс на вкладке поиска. Для экспертизы и печати:

- PDF-выдержки + ссылки + % соответствия  
- **Без:** summary, МКБ-блока, red flags, дифференциала, интерпретаций сервиса, кнопки разбора  

### 1.3 Режим методиста (`?mode=methodist`)

**Задача:** разметить ошибки retrieval → `retrieval_fix` → улучшение RAG / eval.

| Элемент | Поведение |
|---------|-----------|
| Выдача | **компактная**, как у аудита + **оценка ИИ** по каждому протоколу |
| Скрыто | summary, МКБ, red flags, дифференциал, follow-up, разбор/KZ, TOC summary |
| Панель внизу | только форма `retrieval_fix`: top (select), правильный PDF (autocomplete), теги, комментарий |
| Убрано (r129) | дублирующий read-only список путей retrieval — top виден в выдаче выше |

Статистика по исправлениям — **отдельная вкладка** (см. §5), не смешивается с формой разметки.

---

## 2. Текущая архитектура (as-is после r129)

```text
Запрос + МКБ + рубрики
  → POST /api/assist
  → ICD pipeline → retrieve (BM25 + routing + embed rerank)
  → dedupe по basename PDF (1 фрагмент / 1 файл)
  → Gemini JSON (protocols[], summary, differential)
  → confidence_score = blend(LLM, rag_support)
  → UI: compact list | methodist panel | optional detail по кнопке
```

| Слой | Файлы |
|------|-------|
| UI | `index.html` — `runAssist`, `renderLlmOut`, `#search-methodist-panel` |
| API | `rag_server.py` — `/api/assist`, dedupe, `/api/methodist/*` |
| RAG | `retrieve()`, `RAG_MAX_CHUNKS_PER_BASENAME=1` |
| Feedback | `feedback_store.py` — `retrieval_fix`, `source: protocol_search_ui` |
| Телеметрия | `search_telemetry.py`, вкладка «Статистика» |
| ML stats КЗ | `methodist_stats.py`, `GET /api/methodist/stats`, дашборд ML |

---

## 3. План улучшений

### P0 — UX и dedup ✅ (r128–r129)

| ID | Задача | Статус |
|----|--------|--------|
| P0.5 | Dedup PDF по basename; оценка ИИ; разбор/KZ по кнопке | ✅ r128 |
| P0.6 | Режим «только цитаты» | ✅ r128 |
| P0.7 | Компактная выдача в режиме методиста; упрощённая панель `retrieval_fix` | ✅ r129 |
| P0.8 | Lite assist (B1): быстрый JSON только protocols; skip specialty infer при МКБ/рубриках | ✅ r131 |
| P0.9 | Doctor UI (A1): без summary/differential/ICD-блока; match_reason только top-1 | ✅ r131 |

### P1 — качество retrieval (2–4 недели)

| ID | Задача | Приоритет |
|----|--------|-----------|
| P1.1 | Summary-first retrieval при явном МКБ (`condition` → PDF evidence) | высокий |
| P1.2 | Pre-filter рубрика + МКБ до embed rerank | высокий |
| P1.3 | Очередь плохих поисков: `GET /api/methodist/queue?domain=search` | средний |
| P1.4 | Golden queries для поиска (symptom / МКБ / mixed) + CI eval | средний |
| P1.5 | Обязательный `chosen_path` при тегах `wrong_protocol` / `missed_protocol` | средний |
| P1.6 | Hit@1 / Hit@3 на накопленных `retrieval_fix` в дашборде | средний |

### P1-S — статистика поиска и исправлений (как у КЗ)

Цель: зеркалировать контур КЗ — **накопление → дашборд → пороги ML → эффект правок**.

| ID | Задача | Аналог в КЗ | Статус |
|----|--------|-------------|--------|
| S1 | Агрегат `aggregate_search_retrieval_fix()` в `search_telemetry.py` или `methodist_stats.py` | `retrieval_fix` в `build_methodist_dashboard_stats` | ⬜ |
| S2 | KPI на вкладке «Статистика»: число `retrieval_fix` с `source=protocol_search_ui`, Hit@1/3, теги wrong/missed | KPI КЗ: reviews, compliance % | частично (общий `retrieval_fix_count`) |
| S3 | Графики: исправления по дням, по рубрикам rejected/chosen, top query patterns (без текста запроса — только hash/длина) | activity_by_day, rubric_kz_runs | ⬜ |
| S4 | Таблица «последние retrieval_fix» (read-only, без query text) для методиста | re-analysis deltas в ML dashboard | ⬜ |
| S5 | Секция в `GET /api/methodist/stats`: **domain=search** — отдельно от КЗ | protocol_match block | ⬜ |
| S6 | Эффект правок: A/B eval до/после deploy (golden search set) | re-analyze after engine fix | ⬜ |
| S7 | Пороги readiness: ≥20 search `retrieval_fix` для расширения golden; ≥50 для LoRA | `ML_READINESS_THRESHOLDS` | документировано |

**Минимальный MVP статистики (S1+S2+S5):** один экран «Поиск · разметка» в дашборде методиста:

- всего поисков (telemetry)  
- всего `retrieval_fix` (все источники / только search UI)  
- Hit@1, Hit@3 по размеченным парам  
- histogram тегов: wrong_protocol, missed_protocol, query_too_vague  
- прогресс к порогу 20 / 50 для ML  

### P2 — корпус и ML (после ≥20 search retrieval_fix)

- LoRA embedder на `retrieval_pairs_resolved.jsonl`  
- A/B summary-first vs legacy  
- Массовая валидация Protocol Summary  

### P3 — навигация

- Маршрут: жалобы → МКБ → протокол → раздел → цитата PDF  
- Сценарии методиста для ординаторов  

---

## 4. API

### Существующие

- `POST /api/assist` — поиск + dedup в ответе  
- `POST /api/ml/feedback` — `retrieval_fix` с `source: protocol_search_ui`  
- `GET /api/methodist/protocol-search?q=` — autocomplete  
- `GET /api/methodist/stats` — дашборд (сейчас акцент на КЗ; расширить §search)  
- `GET /api/analytics/public` — телеметрия поиска + КЗ (вкладка «Статистика»)

### Планируемые

- `GET /api/methodist/stats?domain=search` — KPI только по поиску  
- `GET /api/methodist/queue?domain=search` — очередь на разметку  

---

## 5. UI: форма методиста на поиске (r129)

После `runAssist` в `body.methodist-mode`:

1. Select «Система ошибочно выбрала» — уникальные PDF из выдачи (короткое имя файла).  
2. Autocomplete «Правильный протокол» — `GET /api/methodist/protocol-search`.  
3. Теги: `wrong_protocol`, `missed_protocol`, `wrong_population`, `query_too_vague`.  
4. «Сохранить retrieval_fix».

Статистику смотреть на вкладках **«Статистика»** (публичная) и **«ML / обучение»** (методист).

---

## 6. Метрики успеха

| Метрика | Сейчас | Цель P1 |
|---------|--------|---------|
| `retrieval_fix` (все источники) | ~9 | ≥20 |
| `retrieval_fix` с search UI | мало | ≥10 |
| Hit@3 на размеченных парах | низкий | ≥60% |
| Доля поисков с МКБ (telemetry) | — | рост |
| Дубликаты PDF в top-5 | 0 после dedup | 0 |

---

## 7. Связанные документы

- `docs/methodist-workbench-tz.md`  
- `docs/methodist-ml-priority-plan.md` — опора A (RAG)  
- `docs/action-plan-master.md` — ML backlog, фазы B–D  
- `clinical_knowledge/search_analytics_public.py` — публичная аналитика  

---

## 8. История версий

| BUILD | Изменения |
|-------|-----------|
| r127 | P0.1–P0.3: methodist panel, summary nav API |
| r128 | Dedup PDF, compact assist, citations-only, tests |
| r129 | Methodist compact UI, упрощённая панель retrieval_fix, roadmap §stats |
