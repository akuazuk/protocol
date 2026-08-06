# МО: полные названия КП + поиск в разборе случая (v1)

Дата: 2026-08-06  
Статус: active  
Связанные: `2026-08-05-mo-case-protocol-suggest-v1.md`,
`2026-08-06-mo-case-findings-clarity-v1.md`,
`2026-08-05-mo-august-llm-bi-backfill-v1.md`.

---

## 1. Контекст

### LLM «вчера» (2026-08-05)

Баннер `llm_queue_pending (80)`: queue=80, grades файл есть, но **все 80 строк с ошибкой**
`User location is not supported` (Gemini geo-block на Render Oregon).
Pending не сбрасывается, пока нет успешных grades + recompute report.

### Подбор КП в «Разборе случая»

Симптомы:

- обрезанные названия (`клинический протокол «Диагностика и лечение`);
- «Открыть КП» / слабые specialty-матчи по 17 баллов;
- reasons «Учитывает замечание: Критический дефект по №55» без связи с КП;
- выдача нерелевантна (уролог → общие «Диагностика и лечение»).

Корни:

1. В `protocol_cards.jsonl` title часто OCR/первая строка без полного названия;
   suggest не использует `protocol_display_name` / имя файла PDF.
2. `specialty_slug=None` в `match_protocol_cards`.
3. Gaps = все findings, включая D_reg55 / template - лепятся в reasons без ранжирования.
4. Кнопка «Открыть КП» = viewer; нужно: **название = прямая ссылка**, кнопка = **поиск по каталогу**.

---

## 2. Цель и метрики

| Метрика | Было | Цель |
|--|--|--|
| `llm_queue_pending` за 2026-08-05 | 80 (все grades error) | 0 после local VPN grade + publish/recompute |
| Длина/читаемость title в suggest | обрезок ≤74 | полное имя из PDF/filename |
| Title кликабелен | нет | да → `/proto-viewer.html` |
| «Открыть КП» | viewer (часто битый/дубль) | поиск каталога с запросом случая |
| Reasons с №55 / template | да | нет (только клинические gaps) |
| Hit specialty+ICD | specialty-only мусор | prefer ICD/clinical; specialty boost не единственный |

---

## 3. Шаги

### A. LLM catch-up 2026-08-05

- [ ] A1. Local `grade_kz_llm.py` с VanyaVPN (Render Gemini blocked) - **blocked**: Render `User location is not supported`; local SSL/IPv6 к Gemini
- [ ] A2. rsync grades → Render secure_cases
- [ ] A3. `recompute_mo_days.py` на Render / local+publish
- [ ] A4. Smoke: yesterday advisory без pending или pending=0

### B. Suggest titles + ranking

- [x] B1. `suggest_title_ru()`: filename/beautify, если registry title truncated/generic
- [x] B2. `specialty_to_rubric` → передавать slug в match (с fallback без фильтра)
- [x] B3. Gaps-фильтр: только клинические; не D_reg55*, E_template*
- [x] B4. Не ставить specialty-only в top-3, если есть ICD-матчи
- [x] B5. `viewer_url` + `search_url` + `search_query` в API

### C. UI

- [x] C1. Title = ссылка на viewer
- [x] C2. Кнопка «Открыть КП» → search_url (`/doctor/search?q=`)
- [x] C3. Доп. кнопка «Карточка» на viewer

### D. Поиск каталога (methodist_protocol_search)

- [x] D1. Полные `protocol_display_name` в выдаче
- [x] D2. Токены/МКБ; limit до 30; viewer_url в items
- [x] D3. Тесты suggest

---

## 4. Риски

- Local Gemini всё ещё может падать без VPN - держать `ensure-on`.
- Жёсткий filter по specialty_slug опустошит выдачу - boost, не exclusive filter.
- Старые карточки без нормального filename - fallback на registry + path.
