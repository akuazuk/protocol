# МО: кабинет врача-эксперта + ID врача в отчётах (v1)

Дата: 2026-08-05  
Статус: active  
Связанные: `2026-08-05-mo-methodist-review-pack-v1.md` (пакеты разбора / gold),
`2026-07-28-mo-daily-bi-platform-v1.md` (CRM SQLite).

---

## 1. Контекст

На Render часто нет ФИО/филиала в витрине (пустой join при ingest / sanitize версий).
Живой MariaDB с Render недоступен. Нужно:

1. Показывать **ID врача**, если есть.
2. Отдельный вход **врача-эксперта** (логин/пароль): отчёт за вчера + все дневные отчёты.
3. Всё, что эксперт открывает и проверяет, сохранять и потом использовать для улучшения оценок.

---

## 2. Быстрый фикс UI: ID врача

Источник на диске Render: `secure_cases/.../mo_YYYY-MM-DD.csv` поля
`doctor_id` (слот 7 протокола) и/или `specialist_id_from_visit`.

| Поле в API | Отображение |
|--|--|
| `doctor_id` | приоритет |
| `specialist_id_from_visit` | fallback |
| `doctor_fio` | если есть - как сейчас; иначе `ID врача: {id}` |

Не ходить в MIS. Обогащение как у `patient_id` (day map из CSV).

---

## 3. Кабинет эксперта - рекомендуемая архитектура (эффективно)

### Не делать

| Вариант | Почему нет |
|--|--|
| Отдельный микросервис / другое приложение | дубли UI, деплой, auth |
| BigQuery / GCP как OLTP для кликов эксперта | латентность, ПДн, ops |
| Live join к MariaDB с Render | нет VPN/секретов; не тот контур |
| Общий `METHODIST_TOKEN` «просто другой PIN» | нет аудита кто именно разметил; нельзя отозвать одного эксперта |
| Писать решения в МИС | чужой прод, нет схемы обучения |

### Делать (минимум нового кода)

```text
[логин эксперт] → session token
    → UI: только «Вчера» + «Отчёты» + fullscreen case
        → open case → access_log / expert_event
        → save decision → crm_review_pack (actor=expert_login, source=expert)
            → позже: gold export → калибровка LLM/scorer
```

Переиспользуем уже сделанное:

- экраны «Вчера» / «Отчёты» / case workspace из `mis-kz-quality.html`;
- таблицу **`crm_review_pack`** (снимок МО + system/LLM + вердикт) - это и есть gold;
- SQLite warehouse на Render disk.

---

## 4. Auth: логин / пароль

### Рекомендация v1

Таблица `crm_expert_user` в том же SQLite warehouse:

```sql
CREATE TABLE IF NOT EXISTS crm_expert_user (
  expert_id TEXT PRIMARY KEY,
  login TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,   -- bcrypt
  display_name TEXT,
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crm_expert_session (
  session_id TEXT PRIMARY KEY,
  expert_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  last_seen_at TEXT
);
```

API:

- `POST /api/expert/login` `{login, password}` → httpOnly cookie / token;
- `POST /api/expert/logout`;
- все `/api/methodist/mo/...` для эксперта через роль `expert` (новый `_mo_role`).

Сиды: 1-3 эксперта через CLI `scripts/mo_expert_user_upsert.py` (пароль не в git).
Опционально bootstrap из env `MO_EXPERT_BOOTSTRAP_JSON` только на первом старте.

### Почему не JWT в localStorage

Cookie httpOnly + server session в SQLite проще отозвать и не светит токен в JS.
Срок сессии: 12ч, sliding на activity.

---

## 5. Права роли `expert`

| Раздел | expert | methodist |
|--|--|--|
| Отчёт за вчера | да | да |
| Список всех дневных отчётов | да | да |
| Открыть случай / МО / PDF | да | да |
| Сохранить review pack | да | да |
| Полный BI (месяц, врачи, heatmap…) | нет v1 | да |
| Bulk CRM / admin / access-log | нет | lead/admin |
| patient_id | да (нужен для разбора) | да |

Реализация: расширить `build_mo_capabilities(role)` + скрыть nav в UI для `expert`.
Отдельный URL входа: `/methodist/expert.html` (тонкая оболочка той же `mo-app.js`
с `forcedPages = yesterday|reports|documents|queue` или deep-link только на эти экраны).

---

## 6. Что сохранять для обучения (критично)

### Уже есть - главный store

`crm_review_pack` на каждое «Сохранить пакет»:

- clinical snapshot;
- system/LLM snapshot;
- decision (3 вердикта + summary + training_use).

Для эксперта писать те же пакеты с:

```json
"actor": "expert:ivanova",
"decision.source": "expert",
"training_use": true
```

### Добавить тонкий audit открытий

Таблица `crm_expert_event` (или переиспользовать `access_log`):

| event | зачем |
|--|--|
| `case_opened` | что смотрели, даже без сохранения |
| `report_opened` | какой день отчёта |
| `review_pack_saved` | уже есть в `crm_case_event` |
| `pdf_opened` | опционально |

Без автосохранения всего текста МО на каждый клик - текст уже в pack при save;
на open достаточно ids + timestamp (дешевле, меньше ПДн-дублей).

### Путь улучшения оценок (после накопления)

1. Фильтр packs: `training_use=1 AND decision.source IN ('expert','methodist')`.
2. Nightly JSONL/Parquet → `medical_exams/gold_review/` (не в git).
3. Eval: LLM KPI / scorer vs `corrected_scores` и verdicts эксперта (precision по 3 вопросам).
4. BigQuery - только если понадобится аналитика объёма; **не** primary.

Цель v1: ≥50 размеченных packs от эксперта за 2 недели пилота.

---

## 7. Метрики

| Метрика | Было | Цель v1 |
|--|--|--|
| ID врача в очереди/карточках | нет | показывается если есть в CSV |
| Отдельный login эксперта | нет | 1+ активный user |
| Экран «Вчера» + отчёты для эксперта | только methodist token | да, role=expert |
| Сохранённые packs с source=expert | 0 | растёт ежедневно |
| BigQuery OLTP | - | нет |

---

## 8. Шаги

- [x] E0: enrich `doctor_id` / `specialist_id` из secure CSV; UI `ID врача: …` при пустом FIO.
- [x] E1: схема `crm_expert_user` + `crm_expert_session` + CLI upsert.
- [x] E2: API login/logout; роль `expert` в capabilities.
- [x] E3: UI `/methodist/expert` - вчера + отчёты с 2026-08-01 + case workspace (reuse).
- [x] E4: review pack save с `source=expert`; events open/save в access_log.
- [ ] E5: smoke на Render; 1 тестовый эксперт; пароль только в Render env.
- [ ] E6 (позже): export gold JSONL + eval harness vs LLM.

---

## 9. Риски

| Риск | Митигация |
|--|--|
| Слабые пароли | bcrypt + мин. длина; ротация CLI |
| Утечка ПДн через эксперта | только авторизованный контур; access_log; нет публичных отчётов |
| Путаница methodist vs expert packs | поле `source` + actor prefix |
| Publish затирает CRM | `crm_*` уже в CRM_TABLES; не трогать при fact-publish |
| Эксперт ждёт полный BI | явно ограничить scope v1 |

---

## 10. Definition of Done v1

1. В очередях/карточках виден ID врача, если есть в secure CSV.
2. Эксперт логинится логином/паролем и видит «Вчера» + список отчётов.
3. Сохранение разбора пишет `crm_review_pack` с actor эксперта.
4. Открытия кейсов пишутся в audit.
5. Путь к gold/eval описан; BigQuery не обязателен.

---

## 11. Порядок работ

1. E0 сейчас (малый PR) - ID врача.
2. Согласовать: 1 тестовый логин, какие дни отчётов (все / только после даты X).
3. E1-E4 одним PR на Render.
4. Пилот 1-2 недели → E6.
