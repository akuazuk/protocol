# План улучшений проекта Protocol

Документ - результат сквозного аудита репозитория (бэкенд, фронтенд, корпус-пайплайн, тесты/CI, безопасность, деплой). Цель - дать приоритизированный, реализуемый план по всем областям с привязкой к конкретным файлам и функциям.

Обозначения приоритета: 🔴 критично, 🟠 важно, 🟡 желательно.

Масштаб кодовой базы (ориентир):
- `rag_server.py` - ~5960 строк (FastAPI, RAG, ICD-пайплайн, consult-review, ~15 промптов, 12 маршрутов).
- `index.html` - ~10 700 строк (CSS + JS + HTML в одном файле, весь JS в одном IIFE на `var`).
- `corpus_pipeline/` - пайплайн извлечения корпуса; параллельно существует legacy-контур (`extract_corpus.py`, `build_chunks.py`).
- `tests/` + `eval/` - локальный каркас QA; CI отсутствует.

Что уже сделано хорошо (база, на которую опираемся):
- Фоновая загрузка корпуса с `_require_rag_loaded()` и 503 до готовности (учёт Render health-check).
- Таймауты вызовов модели через `ThreadPoolExecutor` + futures timeout.
- Лимиты Pydantic на длину запроса и размер/число PDF в consult-review.
- Ключ API хранится только на сервере, в браузер не передаётся.
- Golden-наборы и `eval/run_all.sh` для регрессии поиска.
- `escapeHtml`/`escapeHtmlAttr` в основных рендерах LLM/КЗ, таблица критериев через `textContent`.

---

## 1. Безопасность

### 🔴 1.1 Нет аутентификации и rate-limiting
- Где: все `@app.get/@app.post` в `rag_server.py` (~4771-5915), особенно дорогие `POST /api/assist`, `POST /api/consult-review`, `POST /api/protocol-practical`.
- Риск: любой клиент исчерпывает квоту модели и CPU/RAM; нет защиты от перебора.
- Решение:
  - Ввести API-токен (заголовок `X-API-Key`) через зависимость FastAPI `Depends(verify_token)`; токен из env.
  - Добавить rate-limiting (`slowapi` или собственный лимитер по IP/токену) на тяжёлые маршруты.
  - Раздельные лимиты: поиск (мягко), consult-review (жёстко, дорогой).

### 🔴 1.2 `GET /api/verify-key` публично тратит квоту
- Где: `rag_server.py:4901-4916`, `gemini_verify.py:50-118` - каждый вызов делает реальный `generate_content`.
- Решение: закрыть токеном/админ-ролью; кэшировать результат на N минут; не вызывать модель чаще раза в интервал.

### 🔴 1.3 CORS `allow_origins=["*"]`
- Где: `rag_server.py:4563-4568`.
- Решение: список разрешённых origin из env (`ALLOWED_ORIGINS`), по умолчанию - домен приложения; `allow_credentials` только при явном списке.

### 🔴 1.4 `StaticFiles(directory=ROOT)` монтирует весь репозиторий
- Где: `rag_server.py:5918-5946`.
- Риск: по `/` отдаются `rag_server.py`, `data/`, `clients_consult/*.pdf`; при ошибке конфигурации - потенциально служебные файлы.
- Решение: вынести фронтенд в отдельную папку (`static/` или `public/`), монтировать только её; отдавать `index.html` явным маршрутом.

### 🔴 1.5 ПДн в consult-review
- Где: `rag_server.py:5570-5915`, извлечение демографии `1267-1338`.
- Риск: PDF с персональными данными уходят во внешнюю модель; в ответе - дата рождения, демография; нет политики хранения/удаления.
- Решение:
  - Опция обезличивания текста перед отправкой в модель (маскирование ФИО/дат/номеров).
  - Не сохранять загруженные PDF на диск; явно документировать «в память, без персистентности».
  - Конфиг «контур обработки» (внешний API vs on-premise) и предупреждение в UI.
  - Журнал доступа (audit log) без самих ПДн.

### 🟠 1.6 Prompt injection
- Где: `rag_server.py:5110-5116` (assist), `2119-2127` (extract), `3560-3568` (consult synth).
- Решение: оборачивать пользовательский ввод в явные разделители, инструктировать модель игнорировать инструкции внутри данных, валидировать, что ответ - JSON по схеме.

### 🟠 1.7 Валидация PDF
- Где: проверка только по расширению `.pdf` (`5610-5615`); нет лимита числа страниц (`extract_pdf_text_from_bytes` `3470-3507`).
- Решение: проверка magic bytes (`%PDF-`), лимит страниц, защита от «PDF-бомб».

### 🟡 1.8 Security-заголовки и утечки в ответах
- Нет CSP, X-Frame-Options, HSTS на `FileResponse`/API.
- `/health` и `/api/corpus-stats` отдают `rag_load_error` с внутренними путями/сообщениями (`4771-4881`).
- Решение: middleware с security-заголовками; убрать детали ошибок из публичных ответов (только обобщённый статус).

---

## 2. Бэкенд: архитектура

### 🟠 2.1 Монолит `rag_server.py` (~6000 строк)
- Решение: разнести по модулям без смены поведения:
  - `app/prompts.py` - все строковые промпты (`203-465`).
  - `app/llm.py` - `get_gemini`, `generate_*`, извлечение текста, таймауты.
  - `app/retrieval/` - `retrieve()`, BM25, embed-rerank, токенизация.
  - `app/icd/` - ICD-пайплайн (`_refine_icd_*`, `analyze_query_for_icd`).
  - `app/routes/` - маршруты по доменам (search, consult, icd, meta).
  - `app/schemas.py` - Pydantic-модели.
  - `app/corpus.py` - загрузка/состояние корпуса.

### 🟠 2.2 Побочные эффекты при импорте
- Где: `rag_server.py:4557-4561` - запуск фонового потока загрузки до создания `app`; при `--workers N` каждый воркер грузит корпус отдельно; импорт в тестах запускает загрузку.
- Решение: перенести запуск загрузки в `lifespan`/`startup` event; для многоворкерного режима - общий префлайт или shared-кэш.

### 🟠 2.3 Глобальное mutable-состояние без блокировок
- Где: `_chunks`, `_bm25_index`, `_model`, `_routing`, `_retrieval_embed_meta` (`76-86`, `605-667`, `2517-2536`).
- Решение: инкапсулировать в класс-контейнер; защитить запись `_model`; убрать глобальный `_retrieval_embed_meta` (см. 3.5).

### 🟡 2.4 Дубли и связность
- `_extract_gemini_text` (`2539-2556`) дублирует `gemini_verify._extract_text` (`11-26`).
- `tokenize_ru` в `rag_server.py` импортируется в BM25 - связность retrieval с монолитом.
- Длинные функции: `extract_clinical_detail` (~200 строк, `2072+`), `api_assist` (~315 строк, `4993-5307`), `api_consult_review` (~345 строк, `5570-5915`).
- Решение: вынести общие утилиты; декомпозировать функции на шаги.

---

## 3. Бэкенд: производительность

### 🟠 3.1 `retrieve()` - O(N) по всему корпусу
- Где: `rag_server.py:2348-2407`; в consult-review вызывается 3-4 раза (`5729-5803`).
- Решение: inverted index (токен → список чанков) или предварительный лексический префильтр; кандидатный пул вместо полного прохода.

### 🟠 3.2 Чтение env на каждый вызов
- Где: `retrieve()` читает ~20 переменных окружения за вызов (`2303-2483`).
- Решение: загрузить конфигурацию один раз при старте в dataclass `Settings`; переопределение только через рестарт.

### 🟠 3.3 Эмбеддинги на лету
- Где: `_gemini_embed_rerank_pool` (`849-903`) - до `pool_n` (44) синхронных вызовов на каждый `retrieve()`; `genai.configure()` вызывается заново.
- Решение: использовать офлайн-эмбеддинги чанков (`build_semantic_embeddings.py`) - предрасчёт и хранение; rerank по предвычисленным векторам; эмбеддить только запрос.

### 🟠 3.4 `/health` читает весь ICD JSON
- Где: `rag_server.py:4857-4861` - парсинг `icd10_ru_mkb10su.json` на каждый вызов.
- Решение: считать число записей один раз при старте, кэшировать.

### 🔴 3.5 Гонка `_retrieval_embed_meta`
- Где: `2428-2463`, читается в `5299-5301` - глобал перезаписывается в `retrieve()`; при параллельных запросах метаданные путаются между клиентами.
- Решение: возвращать meta из `retrieve()` как часть результата, не через глобал.

### 🟡 3.6 Накладные расходы потоков
- Где: новый `ThreadPoolExecutor(max_workers=1)` на каждый вызов модели (`2593-2714`).
- Решение: единый общий executor; или нативный async-клиент модели.

### 🟡 3.7 Нет общего дедлайна запроса
- Где: `/api/assist` worst-case - до 8-10 последовательных вызовов модели (`4993-5255`).
- Решение: общий бюджет времени на запрос; ранний выход с частичным результатом.

---

## 4. Бэкенд: надёжность и обработка ошибок

### 🔴 4.1 Блокировка event loop
- Где: `async def api_consult_review` выполняет синхронные `retrieve()`/`generate_gemini()` (`5570-5889`).
- Решение: сделать функцию `def` (FastAPI отправит её в threadpool) или вынести тяжёлую работу в `run_in_executor`.

### 🟠 4.2 Нет retry на 429/quota в основном пути
- Где: `generate_gemini` (`2591-2601`, `5122-5128`); сравн. `gemini_verify.py:99-106`.
- Решение: распознавать 429/RESOURCE_EXHAUSTED, экспоненциальный backoff, понятная ошибка клиенту.

### 🟠 4.3 Хрупкий `int(os.environ.get(...))`
- Где: ~40 мест в `rag_server.py` (напр. `2304`, `2437`, `5117`).
- Решение: хелпер `env_int(name, default)` с `try/except` и логом; при кривом значении - default, не 500.

### 🟠 4.4 Несогласованный контракт ошибок
- `extract_clinical_detail` возвращает `{"error": ...}` в теле 200 (`2139-2142`).
- Retry в assist глотает HTTPException (`5160-5163`).
- Consult second pass пишет ошибку в diag, ответ 200 (`5821-5823`).
- Решение: единый формат ошибок; явные статусы; не маскировать сбои в 200.

### 🟠 4.5 `get_gemini()` без safety_settings
- Где: `2516-2536` vs `gemini_verify.py:76-94` - разное поведение блокировок prod vs verify.
- Решение: общий конструктор модели с едиными safety-настройками.

### 🟡 4.6 Тихая потеря данных
- Битые строки JSONL пропускаются без метрики (`525-528`).
- При `SystemExit` в фоновом потоке сервер «живой», но все RAG-маршруты в 503 (`676-700`).
- Решение: счётчики skip/ошибок в `/health`; healthcheck должен отражать фатальную ошибку загрузки.

---

## 5. Бэкенд: API-дизайн

- 🟡 5.1 Нет версионирования (`/api/v1`); `FastAPI(version="1")` - только метаданные.
- 🟡 5.2 Нет `response_model` ни на одном маршруте - слабая OpenAPI-схема.
- 🟠 5.3 `POST /api/icd-suggest` по факту требует ключ и модель, хотя docstring говорит «без LLM»; не вызывает `_require_rag_loaded()` (`5409-5415`, `4919-4961`).
- 🟠 5.4 Несогласованная нормализация path: `api_protocol_detail`/`api_kz_matrix` не используют `_normalize_protocol_path_key` (`5315-5316` vs `4036-4048`) - URL-encoded path даёт 404.
- 🟡 5.5 `category_slugs`: в assist - JSON-массив, в consult-review - строка `Form` - разные контракты.
- 🟡 5.6 `llm_text` дублирует `llm_json` в ответе assist (`5291`) - лишний payload, «сырой» вывод модели наружу.
- Решение: добавить `response_model`, единый контракт, нормализацию path везде, привести `icd-suggest` в соответствие документации.

---

## 6. Конфигурация и деплой

- 🟠 6.1 `.env.example` (~120 строк): дубли (`RAG_EXTRACT_FULL_MATCH_MAX_CHARS` на стр. 35 и 38, `RAG_EMBED_POOL` 59-60); недокументированные используемые переменные (`RAG_ICD_PRE_RETRIEVE_INFER`, `RAG_KZ_MATRIX_*` и др.). Решение: сгруппировать по разделам, убрать дубли, описать все используемые.
- 🟠 6.2 `render.yaml`: нет `healthCheckPath`, persistent disk закомментирован, нет рекомендаций по RAM/воркерам - риск OOM на 512Mi при полном корпусе. Решение: добавить healthcheck, рекомендации по плану/`RAG_MEMORY_SAVER`.
- 🟠 6.3 Зависимости только нижними границами (`>=`), без lock-файла; `pymupdf` 1.23 vs 1.24 в разных requirements. Решение: lock-файл (`pip-tools`/`uv`), единая версия pymupdf, верхние границы для критичных пакетов.
- 🟡 6.4 `requirements-corpus-pipeline.txt` не входит в основной `requirements.txt` - легко собрать RAG без зависимостей пайплайна.

---

## 7. Корпус-пайплайн и данные

### 🔴 7.1 Два параллельных контура корпуса
- (A) `extract_corpus.py` → `corpus.json` → `build_chunks.py`/`build_structured_index.py`; (B) `corpus_pipeline.run_pipeline` → `chunks.jsonl`.
- RAG читает JSONL, но подмешивает устаревший `structured_index.json` из контура (A) (`rag_server.py:2102-2112`, `4284`).
- Решение: выбрать один контур (рекомендуется `corpus_pipeline`); перестать питать RAG устаревшим индексом; задокументировать единый runbook.

### 🟠 7.2 RAG отбрасывает поля сущностей из чанков
- Где: `_load_chunks_from_jsonl` оставляет только `text`/`page_*`/`chunk_id`/`chunk_type` (`535-574`); `icd10_codes`, `population`, `drugs`, `section_path` теряются.
- Решение: использовать сущности в ранжировании/фильтрах (буст по МКБ, фильтр по разделу).

### 🟠 7.3 МКБ в корпусе ≠ МКБ в runtime
- `corpus_pipeline/entities_extract.py` - упрощённый regex (латиница, без валидации); `icd_mkb.py` (нормализация, кириллические look-alike, валидация по справочнику) не импортируется в пайплайн.
- Нет `icd10_ru_mkb10su.json` в репозитории (только WHO JSON).
- Решение: подключить `icd_mkb` к извлечению в корпусе; валидировать коды по справочнику; зафиксировать источник русского справочника.

### 🟠 7.4 Надёжность пайплайна
- Не идемпотентен, нет resume/checkpoint: один битый PDF → пересбор с нуля (`run_pipeline.py:98-106`).
- Битый PDF пропускается без записи в `failed.json`/registry.
- Нет валидации выхода (уникальность `chunk_id`, минимальная длина, согласованность `chunk_count`).
- Логи - только `print`/stderr, без уровней и счётчиков.
- Решение: кэш по mtime/hash, resume, манифест ошибок, JSON-валидация выхода, структурное логирование.

### 🟠 7.5 Качество извлечения
- `page.get_text("text")` без блоков/колонок - ломается порядок в многоколоночной вёрстке (`pdf_extract.py:72`).
- Legacy `extract_corpus` схлопывает весь текст в одну строку (`_WS.sub`) - теряет структуру (`extract_corpus.py:35`).
- OCR опционален и слабый (dpi 150, без препроцессинга), зависимости закомментированы.
- `text_normalize.py` не чинит soft hyphen, переносы слов, NBSP, типографские кавычки.
- Таблицы: первая строка всегда header; `merge_multipage_tables` - заглушка; двойное извлечение без дедупликации; табличные чанки только для первого логического документа.
- Разделы: ложные срабатывания regex; плоский `section_path` без иерархии.
- Решение: извлечение по блокам, восстановление переносов, улучшение таблиц и разделов, дедупликация.

### 🟡 7.6 Структура данных и воспроизводимость
- Два разных `index.csv` (корень vs `output/registry`) без связи `path ↔ doc_id`.
- `split_chunks_jsonl.py` без манифеста (нет sha256, даты, версий) - риск неверного порядка склейки.
- Четыре копии словаря специальностей с расхождениями (`config.py`, `build_protocol_meta.py`, `build_semantic_embeddings.py`, `passport_build.py`).
- `doc_id` = SHA256 пути - переименование PDF меняет id.
- Нет фиксации версии корпуса (hash набора, дата crawl, git commit).
- Решение: единый реестр, манифест корпуса, общий словарь специальностей, стабильный doc_id по содержимому.

---

## 8. Фронтенд

### 🔴 8.1 Старт UI завязан на `protocols.json`
- Где: `index.html:10633-10702` - обработчики формы/рубрик/голоса/навигации вешаются только после успешного fetch.
- Решение: инициализировать UI независимо; данные подгружать прогрессивно; graceful degradation.

### 🔴 8.2 Consult-review без AbortController/таймаута
- Где: `index.html:9919-10003` (в отличие от assist `10527-10531` и protocol-detail `7167-7170`).
- Решение: добавить `AbortController` + таймаут, видимый прогресс и сообщение об ошибке.

### 🟠 8.3 Монолит и стиль JS
- ~10 700 строк в одном файле; весь JS на `var` в одном IIFE; десятки `window.__*` глобалов; дубли init (`initRagBaseInput` вызывается дважды: `10613` и `10658`).
- Решение: вынести CSS и JS в отдельные файлы, затем лёгкая сборка (esbuild/vite); перейти на `const`/`let` и модули поэтапно.

### 🟠 8.4 Доступность
- Вкладки без ARIA-паттерна (стрелки/Home/End, roving tabindex) - `switchMainAppTab` `9435-9456`, только `click`.
- Модалки без focus-trap/Esc/возврата фокуса (`presentation-overlay`, `assist-timer-overlay`).
- Дублирующийся `id="presentation-title"` при каждом `renderSlide` (`9105-9106`).
- Нет `<main>`/skip-link; нет `prefers-reduced-motion` в приложении.
- Решение: реализовать ARIA Tabs, focus-trap, уникальные id, landmark-разметку, reduce-motion.

### 🟠 8.5 Обработка ошибок и UX
- Ошибки через `alert()` (`6634`, `6755`, `10165` и др.); пустые `.catch(()=>{})` (`8954`, `8987`, `9240`, `9287`) - тихие сбои.
- Решение: единый компонент уведомлений в `aria-live`; не глотать ошибки.

### 🟠 8.6 XSS-поверхность
- Ссылки на PDF через `encodeURI(path)` без проверки схемы (`6978-6988`, `9552-9556`) - опасно при компрометации/подмене ответа API (`javascript:`/`data:`).
- `renderSlide`: `innerHTML += s.html` без санитизации (`9105-9106`).
- `restoreAssistUndo`: повторная вставка `innerHTML` без повторного escape (`6092`).
- Атрибуты `id`/`for` из slug без `escapeHtmlAttr` (`5661-5668`).
- Числа из API в `innerHTML` с допущением «всегда число» (`8925-8951`, `9218-9228`, `9371-9372`).
- Решение: валидировать схему URL (только http/https/относительные), централизованный безопасный билдер DOM, escape для всех данных от сервера.

### 🟠 8.7 Производительность фронтенда
- Блокирующие шрифты Google без `display=swap`/`font-display`; внешний логотип `kravira.by` без fallback.
- В презентации Mermaid без `defer` + синхронный `initialize`.
- Таймеры UI 250/450 ms при ожидании модели - лишние перерисовки.
- Решение: `font-display: swap`, локальный fallback логотипа, `defer` для Mermaid, реже обновлять UI-таймеры.

### 🟡 8.8 SEO/мета/i18n
- Нет favicon, Open Graph, `rel=canonical`; `consult_review.html` - редирект без мета.
- Моноязычный (`lang="ru"`), без переключателя.
- Дублирование print-логики между `index.html` и `docs/*`.
- Решение: favicon + OG + canonical; вынести общий print-CSS/JS.

---

## 9. Тестирование, качество, CI/CD

### 🔴 9.1 Нет CI
- `.github/workflows/` отсутствует; `pyproject.toml`/`pytest.ini` нет; `eval/run_all.sh` запускается только вручную.
- Решение: GitHub Actions на push/PR: установка зависимостей → `ruff` → `pytest tests/ -q` → `eval/run_all.sh` (mini).

### 🔴 9.2 Главный путь не покрыт
- `POST /api/assist`, весь PDF-пайплайн, `icd_mkb.py`, фронтенд - без автотестов.
- embed-rerank в тестах отключён (`RAG_GEMINI_EMBED_RERANK=0`) - поведение прода не проверяется.
- Решение: тест `/api/assist` с моком модели (recorded fixtures); unit-тесты `icd_mkb`; тесты `corpus_pipeline` на 1-2 фикстурных PDF; nightly job с реальным ключом и embed-rerank.

### 🟠 9.3 Линтеры и форматирование
- Нет ruff/black/mypy/eslint, нет pre-commit.
- Решение: `ruff` (lint+format) + `.pre-commit-config.yaml`; eslint при появлении JS-сборки.

### 🟠 9.4 Golden и регрессия качества
- Mini-корпус из 2 чанков не отражает ~450 PDF; prod golden (18 запросов) в `.gitignore`.
- `test_retrieve_smoke.py` делает `skip` вместо fail при пустой выдаче.
- Нет порога pass_rate в автоматике.
- Решение: закоммитить subset prod-golden с `expected_any_path_contains`; жёсткий fail вместо skip; порог качества в CI.

### 🟡 9.5 E2E и coverage
- `e2e/` - только README; coverage не настроен.
- Решение: Playwright smoke (ввод запроса → отображение результата); `pytest-cov` с порогом для ключевых модулей.

### 🟡 9.6 Зависимости тестов
- `requirements-dev.txt` минимальный; eval-зависимости неявно.
- Решение: явно зафиксировать dev/eval-зависимости.

---

## 10. Наблюдаемость (observability)

- 🟠 Нет структурного логирования (request_id, latency retrieve/LLM, размер пула embed).
- 🟡 Нет метрик (Prometheus/статистика по маршрутам), нет трейсинга цепочки вызовов модели.
- Решение: добавить `logging` с уровнями и request_id; по возможности - базовые метрики и счётчики ошибок/квоты.

---

## Дорожная карта по фазам

Каждая фаза самодостаточна и заканчивается коммитом + push.

### Фаза 1 - Безопасность (быстрый, высокий эффект) - ВЫПОЛНЕНО
- 1.3 CORS из env (`ALLOWED_ORIGINS`, по умолчанию same-origin) - сделано.
- 1.4 `SafeStaticFiles` - блокировка исходников, конфигов, `data/`, `clients_consult/` (ПДн-PDF) при раздаче по `/` - сделано.
- 1.1 rate-limiting in-memory по IP на дорогих маршрутах (`RATE_LIMIT_*`) - сделано (токен-аутентификация маршрутов оставлена опциональной, чтобы не ломать браузерный фронтенд; для verify-key добавлен опциональный `X-Admin-Token`).
- 1.2 `/api/verify-key`: кэш результата (`VERIFY_KEY_CACHE_TTL`) + опциональный admin-токен - сделано.
- 1.8 security-заголовки (HSTS, X-Frame-Options, nosniff, Referrer-Policy), опциональный CSP, чистка текста внутренних ошибок в публичных ответах (`public_error_text`, `DEBUG_ERRORS`) - сделано.
- Результат: закрыт публичный доступ к файлам репозитория и ПДн-PDF, ограничена нагрузка на дорогие ручки, безопасные заголовки на всех ответах.

### Фаза 2 - Устойчивость - ВЫПОЛНЕНО
- 4.1 `/api/consult-review` переведён на синхронный обработчик (FastAPI выполнит его в threadpool, не блокируя event loop), файлы читаются синхронно (`uf.file.read()`).
- 3.5 устранена гонка `_retrieval_embed_meta` - переведено на `threading.local()` (`_set/_get_retrieval_embed_meta`).
- 4.2 retry при 429/quota с backoff (`_run_model_with_retry`, `GEMINI_QUOTA_RETRY*`), применён к `generate_gemini` и `generate_gemini_plain`.
- 4.5 единые safety-настройки модели в `get_gemini()`.
- 4.3 безопасное чтение env (`env_int`/`env_float`) в горячих путях retrieve и consult-review.
- 1.7 проверка magic bytes PDF и лимит страниц (`CONSULT_REVIEW_MAX_PAGES`).
- 8.2 `AbortController` + таймаут на consult-review (фронтенд) с понятным сообщением.
- 8.1 независимый старт UI - критичная инициализация вынесена из промиса `protocols.json` (главная вкладка и поиск работают, даже если файл не загрузился).

### Фаза 2 - Устойчивость (исходный план) в проде
- 4.1 убрать блокировку event loop в consult-review; 3.5 устранить гонку `_retrieval_embed_meta`; 4.2 retry на 429; 4.3 безопасный `env_int`; 4.5 единые safety-настройки; 8.2 AbortController в consult-review; 8.1 независимый старт UI.
- Результат: стабильное поведение при нагрузке и сбоях модели.

### Фаза 3 - Качество и CI - ВЫПОЛНЕНО
- GitHub Actions CI (`.github/workflows/ci.yml`): ruff + pytest на push/PR в main, Python 3.11.
- ruff сконфигурирован (`ruff.toml`, набор F/E9/W6), исправлены найденные F401/F841.
- `.pre-commit-config.yaml` (ruff + базовые хуки), `ruff`/`pre-commit` добавлены в `requirements-dev.txt`.
- Тест `/api/assist` с мок-моделью (`tests/test_api_assist.py`): валидация 422, путь без ключа 503, успешная сборка ответа.
- Унифицирована версия `pymupdf` (`requirements-search.txt` -> >=1.24.0, как в pipeline).
- Порог качества: агрегаты `summary` в отчёте eval + `eval/quality_gate.py` (gate по `pass_rate`, `QUALITY_MIN_PASS_RATE`).

### Фаза 3 - Качество и CI (исходный план) (защита от регрессий)
- 9.1 GitHub Actions; 9.3 ruff + pre-commit; 9.2 тест `/api/assist` с моком + unit `icd_mkb`; 6.3 lock-файл и единый pymupdf; 9.4 порог качества на mini.
- Результат: каждое изменение проверяется автоматически.

### Исправление воспроизводимости оценки (ВЫПОЛНЕНО)
Симптом: повторная загрузка одного и того же PDF в «Проверке КЗ» давала разный «Ориентировочное соответствие».
Причина: все вызовы модели шли с `temperature>0` (0.25/0.22/0.18/0.15/0.1) без фиксированного seed - модель сэмплировала разные ответы (а изменение digest меняло и запрос RAG, и контекст, и итоговый %).
Исправление:
- единый билдер `_make_generation_config` для всех 6 точек генерации: по умолчанию `temperature=0` (жадное декодирование), `candidate_count=1`, опц. `GEMINI_TOP_P/TOP_K/SEED` (`GEMINI_TEMPERATURE` для возврата вариативности);
- детерминированный tie-break в ранжировании `retrieve` и embed-rerank (по `path` + `chunk_index`) - стабильный порядок при равных score;
- регрессионные тесты `tests/test_determinism.py`.
Результат: одинаковый вход -> одинаковый digest -> один и тот же отбор протоколов -> одинаковый %.

Дополнение (после повторной жалобы «всё ещё разное»): `temperature=0` у Gemini не даёт 100% гарантии
(остаточная недетерминированность на стороне API + `overall_compliance_pct` — свободно генерируемое число).
Поэтому добавлен **кэш результата по контент-хэшу файлов** (`_consult_cache_*`): один и тот же PDF (тот же контент)
+ те же рубрики/модель/настройки -> идентичный результат из кэша, тяжёлый разбор выполняется один раз.
Ключ кэша строится по **нормализованному извлечённому тексту** (`_normalize_for_cache`: схлопывание пробелов,
нижний регистр), а не по сырым байтам PDF: один и тот же по содержанию документ совпадает в кэше, даже если
файл пересохранён/переэкспортирован (другие байты). Состав ключа: SHA-256(нормализованный текст + рубрики +
модель + temperature + embed-настройки + метод overall + версия кэша). Управление:
`CONSULT_REVIEW_CACHE`, `CONSULT_REVIEW_CACHE_MAX`. Тесты: `tests/test_consult_cache.py`.

Версионирование развёртывания: добавлена встроенная `BUILD_VERSION` в `rag_server.py` (используется
`_app_version()` по умолчанию, переопределяется env `APP_VERSION`). Версия видна: в `/api/version`, `/health`,
`/api/corpus-stats`, в ответе `consult-review` (`server_version`) и **в футере сайта** (бейдж «Версия: …»,
`loadBuildVersionBadge` тянет `/api/version`). Так сразу видно, новый код развёрнут на сайте или старый.

Дополнительно: `overall_compliance_pct` теперь считается детерминированно как среднее баллов критериев
(`_stabilize_overall_compliance`, `CONSULT_REVIEW_OVERALL_FROM_CRITERIA`) - итог прозрачен и не зависит
от отдельного «свободного» числа модели. В ответе добавлены поля `cached_result` и `overall_compliance_method`
(по ним легко проверить, что на сервере уже работает новый код).

### Безопасный инкремент фаз 4/6/7 (ВЫПОЛНЕНО, без смены архитектуры)
По решению - только низкорисковые части (без разбиения файлов и lifespan-миграции, чтобы не ломать живое приложение):
- 3.4 кэш числа ICD-записей в `/health` (`_icd_ru_entries_count`).
- 10 observability: логирование запросов и времени (`protocol.rag`), заголовок `X-Process-Time-Ms`, лог 429/500/медленных (`REQUEST_LOG`, `SLOW_REQUEST_MS`, `LOG_LEVEL`).
- Версия сборки: `APP_VERSION` в `/health`, `/api/corpus-stats`, новый `/api/version`.

### Фаза 4 - Рефакторинг архитектуры (ОТЛОЖЕНО, высокий риск)
- 2.1 разнесение `rag_server.py` по модулям; 2.2 загрузка через lifespan; 8.3 вынос CSS/JS из `index.html`; 2.4 устранение дублей; 3.2 единая конфигурация.
- Результат: поддерживаемость и тестируемость. Требует отдельной итерации с расширенным тестовым покрытием.

### Фаза 5 - Данные и пайплайн
- 7.1 единый контур корпуса; 7.3 подключение `icd_mkb`; 7.4 resume + манифест + валидация; 7.5 таблицы/OCR/нормализация; 7.6 манифест версии корпуса.
- Результат: воспроизводимый корпус и качество извлечения.

### Точность анализа КЗ: разметка корпуса + использование структуры в рантайме (ВЫПОЛНЕНО)
Главная проблема была в том, что богатая разметка корпуса (`section_path`, `page_*`, `point_numbers`, `icd10_codes`)
**не доходила** до рантайма и до промпта модели. Сделано двунаправленно:

Разметка корпуса (`corpus_pipeline/`):
- `section_detect.py`: распознавание номера раздела (`section_number`), читаемого `section_title` и
  **иерархического** `section_path` по числовой вложенности (`2` -> `2.1`).
- `tables_extract.py`: реальный `merge_multipage_tables` (склейка таблиц с тем же заголовком через границу
  страниц, расширение `page_from..page_to`) вместо заглушки; `chunk_build.py` пишет `table_title` и диапазон страниц.
- `entities_extract.py`: валидация `icd10_codes` по справочнику ВОЗ (`data/icd_reference`, корни 3 символа),
  отключаемо `CORPUS_ICD_VALIDATE=0`.
- `chunk_build.py`: чанки несут `section_title`/`section_number`; `embedding_ready_text` строится по `section_title`.

Рантайм (`rag_server.py`), всё за обратимыми флагами:
- `_load_chunks_from_jsonl`: сохраняет `section_path`/`section_title`/`point_numbers`/`icd10_codes`/`page_*`
  (`RAG_KEEP_STRUCT=1` по умолчанию; `0` — прежнее поведение).
- `retrieve()`: в результат добавлены `section_title`/`page_from`/`page_to`/`point_numbers` (скоринг не изменён).
- `_build_review_chunks_context` + `SYSTEM_CONSULT_REVIEW_JSON`: в выдержки протоколов добавляются
  `section=`/`pages=`/`пункты=`, модель просят перенести их в `protocol_section`/`protocol_page`
  (`CONSULT_REVIEW_RICH_CONTEXT=1` по умолчанию; `0` — старый формат для сравнения/отката).
- UI: `consult_protocol_fragments` несут `section`/`pages`; `index.html` показывает «Раздел: …, стр. N».

Скачивание (`download_minzdrav_protocols.py`): манифест `minzdrav_protocols/_manifest.jsonl`
(`url`, `sha256`, `bytes`, `downloaded_utc`, `http_status`, `action`), флаг `--refresh` (перекачать и сравнить sha256),
журнал ошибок `_download_errors.json`.

Версия: `BUILD_VERSION` бампнута; `/api/version` теперь отдаёт `corpus_chunks`, `corpus_structured_chunks`,
`keep_struct`, `consult_rich_context` — по ним видно, что корпус структурирован и какой код/настройки на сервере.

Тесты: `tests/test_corpus_structure.py`, `tests/test_runtime_struct.py`, `tests/test_download_manifest.py`.

Пересборка корпуса (запускать на машине с PDF, сетью и `pymupdf`/`pdfplumber`):
```
python3 download_minzdrav_protocols.py --refresh
python3 build_index.py && python3 build_protocol_meta.py
python3 -m corpus_pipeline.run_pipeline
python3 split_chunks_jsonl.py
```

### Фаза 6 - Производительность и доступность (частично ВЫПОЛНЕНО)
- **3.1 inverted index** — `RAG_LEX_INVERTED_INDEX=1` (по умолчанию): токен→чанки при загрузке корпуса, retrieve без полного прохода.
- **3.4 кэш `/health`** — было ранее (ICD count).
- **8.4 a11y** — инкремент в фазе 7 (skip-link, tabs, focus-trap).
- **8.7 фронтенд-перф** — инкремент в фазе 7.
- **Consult SSE** — `POST /api/consult-review/stream`: прогресс в % и partial results (протоколы, правила, МКБ до финальной модели).
- **3.3 офлайн-эмбеддинги (инкремент)** — `RAG_PRECOMPUTED_CHUNK_EMBED=1`: rerank по полю `embedding` в JSONL (один API-вызов на query).
- **Каталог нозологий** — `clinical_knowledge/condition_registry.py`; скрипты `scripts/catalog_rules_coverage_report.py`, `scripts/build_catalog_llm_enrichment.py`.
- **Правила по всему каталогу (r23)** — `clinical_knowledge/catalog_build.py`, `scripts/build_catalog_rules.py` → `data/catalog/rules/` + `rules_coverage_report.json` (478 PDF, 24 рубрики); loader мержит gastro + catalog; rule_checker v3 с runtime path-правилами и `condition_registry`; consult pipeline без жёсткой привязки к одной рубрике (scope `all_catalog`).
- **Полная структуризация как gastro (r25)** — `catalog_full_build.py`, `condition_builder.py`, `scripts/build_catalog_full.py` → `data/catalog/conditions/` (JSON нозологий), прогресс % в CLI/SSE; `/api/clinical-knowledge/build-status`; generic corpus «формулировка диагноза» + path-шаблоны всех рубрик.
- 3.3 полный offline corpus embed (build_semantic_embeddings) — отложено.

### Фаза 7 - Полировка (частично ВЫПОЛНЕНО)
- 8.8 SEO/мета - сделано: robots (noindex для пилота с ПДн), theme-color, OpenGraph/Twitter, инлайн-favicon (`index.html`).
- 6.1 чистка `.env.example` - сделано: убраны дубли (`RAG_EMBED_POOL`, `RAG_EXTRACT_FULL_MATCH_MAX_CHARS`), висячий комментарий; добавлены новые переменные (retry, лимит страниц PDF, observability, версия).
- 10 observability - сделано в безопасном инкременте (см. выше).
- **Render 502 consult-review** - сделано: второй RAG-pass выкл на Render по умолчанию, preflight `/health`, понятные ошибки HTML 502/504 (`rag_server.py`, `index.html`, `render.yaml`).
- **8.4 a11y (инкремент)** - skip-link, клавиатура вкладок (Arrow/Home/End), focus-trap + Esc для presentation-overlay, focus-trap для assist-timer-overlay, `prefers-reduced-motion`.
- **8.6 XSS (инкремент)** - `safeResourceHref()` для ссылок на PDF протоколов.
- **8.7 фронтенд-перф (инкремент)** - fallback логотипа при ошибке загрузки; реже тики таймеров (500/700 ms).
- **9.3** - `pyproject.toml` (pytest + метаданные проекта; ruff остаётся в `ruff.toml`).
- **9.4** - `test_retrieve_smoke.py`: жёсткий fail вместо skip при пустой выдаче.
- 5.x API-дизайн (v1, response_model, нормализация path) - отложено (риск для совместимости с фронтендом).

---

## Быстрые победы (можно сделать в первую очередь, низкий риск)
- Убрать дубли в `.env.example` (6.1).
- `font-display: swap` и favicon (8.7, 8.8).
- `env_int` хелпер и замена хрупких `int(env)` (4.3).
- Кэш числа ICD-записей в `/health` (3.4).
- `ruff` + базовый `pyproject.toml` (9.3).
- Жёсткий fail вместо skip в `test_retrieve_smoke.py` (9.4).
