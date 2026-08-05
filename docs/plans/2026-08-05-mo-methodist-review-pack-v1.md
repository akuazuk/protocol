# МО: идентификаторы в отчётах, fullscreen-разбор, пакет решения методиста (v1)

Дата: 2026-08-05  
Статус: active  
Связанные:

- `2026-07-28-mo-daily-bi-platform-v1.md` - CRM / warehouse;
- `2026-08-05-mo-llm-action-queue-judge-v1.md` - shadow LLM A/B (если есть в индексе / merged);
- `2026-08-04-mo-runtime-stabilization-v1.md` - Docker / worker; GCP как фаза C;
- этот план (§11): целевой хост **GCE VM `europe-north1`** (дешевле/предсказуемее Render для always-on + диск);
- consult feedback JSONL (`clinical_knowledge/feedback_store.py`) - паттерн обучения, не CRM.

---

## 1. Контекст

Методисту нужно:

1. В отчётах и очередях видеть **идентификаторы и оргконтекст** визита.
2. Открывать **«Разбор случая» на весь экран** (сейчас боковой drawer ~760px).
3. Заполнять **развёрнутое решение** и сохранять его **вместе с текстом МО и LLM/scorer-разбором**, чтобы потом:
   - пересмотреть / исправить;
   - выгрузить как gold для улучшения разбора.

Сейчас:

| Что | Как есть |
|--|--|
| CRM | SQLite `crm_case_state` / `crm_case_event`, ключ = `visit_id` |
| Решение | status, assignee, due, tags, finding_decisions; comment только в event payload |
| МО-текст | secure CSV / case document API, не в CRM |
| LLM-разбор | `judges.jsonl` на диске, shadow |
| `patient_id` | есть в secure CSV; **намеренно вычищен** из public API / warehouse facts |
| BigQuery | не как OLTP; опциональный export |
| Хостинг | Render web + Mac/SSH publish | рассмотреть **GCE VM europe-north1, 8 GB RAM** (§11) |

Уточнение по железу: в запросе указано «8 мегабайт» - для web+warehouse это нереалистично. В плане принимаем **8 GB RAM** (типичный `e2-standard-2`: 2 vCPU / 8 GB). Если имелось в виду иное - поправить до закупки VM.

---

## 2. Цель и метрики

| Метрика | Было | Цель v1 |
|--|--|--|
| Поля в action-очереди / таблицах отчётов | visit частично, patient нет, филиал не везде | visit_id, patient_id (роль), дата, ФИО, спец., филиал |
| Drawer | боковая панель | fullscreen workspace |
| Решение методиста | 5 коротких полей | структурированный вердикт + комментарий по 3 вопросам |
| Сохранённый «пакет разбора» | нет | 1 запись на сохранение: meta + clinical snapshot + judge snapshot + decision |
| Primary store | SQLite warehouse | тот же (+ export позже) |
| BigQuery как OLTP | - | **нет** в v1 |
| Хост данных/CRM | Render disk | **v1 DoD = Render only**; GCE `europe-north1` (§11) отложен до отдельного решения |

---

## 3. Рекомендация по хранилищу (критично)

### Не делать BigQuery первичным

BigQuery / «просто SQL в GCP» плохо подходит как рабочая БД CRM:

- высокая задержка записи/чтения для UI;
- нет нормальных транзакций/апдейтов как в OLTP;
- усложняет Render-deploy и доступ методиста;
- ПДн (patient_id, текст МО) требуют отдельного контура и IAM.

GCP (Cloud Run Job + GCS) уже в плане runtime - это про **pipeline**, не про CRM-форму.

### Делать так

```text
[UI fullscreen]
    → POST review-pack
        → SQLite warehouse на persistent disk (Render сейчас / GCE europe-north1 цель)
            → (опционально) nightly export → GCS → BigQuery (аналитика / обучение)
```

**v1 product (A/B) на Render:** схема review pack и UI поверх текущего SQLite API на persistent disk Render.  
**Host / GCE:** перенос на **GCE VM `europe-north1`** (§11) **отложен** - не блокирует A/B; отдельный go/no-go после стабилизации пакетов на Render.  
**v1.5 / v2 analytics:** nightly snapshot JSONL/Parquet → GCS; BigQuery только витрина обучения.

Альтернатива «MariaDB MIS» для хранения решений - **не использовать**: чужой прод МИС, нет схемы под наши пакеты, риск смешения с клинической записью.

---

## 4. Продуктовая модель «пакет разбора» (review pack)

Одна версия пакета = снимок на момент сохранения методистом.

```json
{
  "pack_id": "uuid",
  "case_id": "3646270",
  "visit_id": "3646270",
  "mis_id": "898517",
  "patient_id": "…",
  "visit_date": "2026-08-04",
  "doctor_fio": "…",
  "specialty": "…",
  "filial": "…",
  "document_kind": "consultation",
  "clinical_snapshot": { "complaints": "…", "…": "…" },
  "system_snapshot": {
    "overall_pct": 72,
    "findings": [],
    "llm_action_judge": { },
    "rubric_mz": { }
  },
  "methodist_decision": {
    "status": "confirmed_issue",
    "verdict_completeness": "agree|disagree|partial",
    "verdict_diagnosis": "agree|disagree|partial",
    "verdict_recommendations": "agree|disagree|partial",
    "corrected_scores": { "completeness": 60, "diagnosis": 40, "recommendations": 20 },
    "summary_ru": "свободный разбор",
    "finding_decisions": {},
    "assignee": "",
    "due_date": "",
    "tags": [],
    "training_use": true
  },
  "actor": "methodist@…",
  "created_at": "…",
  "supersedes_pack_id": null
}
```

Правила:

- каждое «Сохранить» = **новая версия** (append), предыдущая не затирается;
- `crm_case_state` остаётся быстрым индексом текущего статуса;
- клинический текст копируется в snapshot (независимо от ротации secure CSV);
- `patient_id` только для ролей methodist/lead/admin; в публичные/doctor API не отдавать;
- audit: `access_log` при просмотре patient_id / пакета.

---

## 5. Фазы реализации

### Фаза A - отчёты + fullscreen + расширенная форма (быстро, 1-2 дня)

**A1. Поля в отчётах / очередях**

Добавить в UI и API (action queue, documents table, daily report items, CSV export где есть):

| Поле | Источник |
|--|--|
| ID визита | `visit_id` / `case_id` |
| ID пациента | join из secure CSV **только methodist+**; иначе «-» / скрыто |
| Дата визита | `visit_date` |
| ФИО врача | `dim_doctor` / `doctor_fio` |
| Специальность | sanitize specialty |
| Филиал | sanitize filial |

Не класть `patient_id` в `fact_mo_case` без отдельного решения по ПДн; для v1 - lookup по secure_cases при сборке daily/case detail для авторизованного методиста.

**A2. Fullscreen «Разбор случая»**

- CSS: drawer `width: 100%`, max-width none, двухколоночный layout (МО | разбор+решение);
- опционально query `?case=` deep-link уже есть - сохранить;
- на мобилке - одна колонка, sticky «Решение».

**A3. Расширенное «Решение методиста»**

Поля сверх текущего:

- согласие с 3 KPI LLM (полнота / диагноз / рекомендации): agree / partial / disagree;
- опционально ручная корректировка % (shadow, не primary warehouse);
- развёрнутый `summary_ru` (textarea 500-4000);
- флаг `training_use` («можно в gold»);
- сохранить finding_decisions как сейчас.

Пока без отдельной таблицы - писать расширенный JSON в `crm_case_event.payload_json` + дублировать ключевые поля в `crm_case_state` через новые колонки или `decision_json`.

### Фаза B - таблица review pack + просмотр/правка истории (2-4 дня)

**B1. Schema**

```sql
CREATE TABLE IF NOT EXISTS crm_review_pack (
  pack_id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL,
  visit_id TEXT NOT NULL,
  mis_id TEXT,
  patient_id TEXT,
  visit_date TEXT,
  doctor_fio TEXT,
  specialty TEXT,
  filial TEXT,
  clinical_json TEXT NOT NULL,
  system_json TEXT NOT NULL,
  decision_json TEXT NOT NULL,
  training_use INTEGER NOT NULL DEFAULT 1,
  actor TEXT,
  created_at TEXT NOT NULL,
  supersedes_pack_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_crm_review_pack_case ON crm_review_pack(case_id, created_at DESC);
```

**B2. API**

- `POST /api/methodist/mo/cases/{id}/review-pack` - сохранить пакет;
- `GET /api/methodist/mo/cases/{id}/review-packs` - список версий;
- `GET /api/methodist/mo/review-packs/{pack_id}` - полная карточка;
- `POST .../review-packs/{pack_id}/revise` - новая версия на базе старой.

При save: подтянуть clinical + llm_action_judge + findings в snapshot атомарно.

**B3. UI**

- вкладка / блок «История разборов» в fullscreen;
- открыть прошлую версию, «Исправить» → prefill формы → новый pack;
- фильтр «gold / training_use» в отчётах.

### Фаза C - обучение / GCP export (позже, по объёму)

Когда накопится ≥100-200 размеченных пакетов:

1. Nightly job: `review_packs` → Parquet/JSONL под `medical_exams/gold_review/` (не в git).
2. Опционально: sync в GCS bucket; BigQuery external table / load для аналитики согласий методиста vs LLM.
3. Eval harness: сравнить LLM KPI с `corrected_scores` / verdicts методиста (precision по 3 вопросам).

Не блокировать A/B ожиданием BigQuery.

---

## 6. UX (целевой экран)

```text
+-- Разбор случая (fullscreen) ----------------------------------------------------------+
| visit_id · patient_id · date · doctor · specialty · filial                   [Close] |
+----------------------------------+-----------------------------------------------------+
| Реальное МО (слоты)              | 3 KPI LLM + блоки полноты + findings                |
|                                  | --------------------------------------------------- |
|                                  | Решение методиста (развёрнуто)                      |
|                                  |  status / assignee / due                            |
|                                  |  Полнота / Диагноз / Рекомендации: agree|partial|.. |
|                                  |  corrected % (optional)                             |
|                                  |  summary_ru (textarea)                              |
|                                  |  [x] training_use (gold)                            |
|                                  |  [Сохранить пакет]                                  |
|                                  |  История версий: v3 <- v2 <- v1                     |
+----------------------------------+-----------------------------------------------------+
```

---

## 7. Шаги (чеклист)

- [x] A1: колонки visit/patient/date/doctor/specialty/filial в очереди и таблицах (+ export).
- [x] A1b: patient_id только methodist+; access_log.
- [x] A2: fullscreen drawer / workspace layout.
- [x] A3: расширенная форма решения + запись в CRM event/state.
- [x] B1: таблица `crm_review_pack` + migrate.
- [x] B2: API save/list/get/revise.
- [x] B3: UI истории и правки.
- [ ] H1: решение по GCE `europe-north1` (тип VM, диск, DNS) - §11 - **deferred**.
- [ ] H2: Dockerfile.web + systemd/compose на VM; persistent disk для `medical_exams` - **deferred**.
- [ ] H3: cutover DNS / домен с Render → VM; rollback план - **deferred**.
- [ ] H4: cron/worker daily на той же VM (или отдельная spot/preemptible) - **deferred**.
- [ ] C: nightly export gold (GCS/BQ опционально).

---

## 8. Риски

| Риск | Митигация |
|--|--|
| ПДн patient_id / текст МО в snapshot | роли, access_log, не в public reports, не в git |
| Рост SQLite | пакеты текстовые; мониторинг размера; C - вынос cold в файлы/GCS |
| Publish затирает CRM | как сейчас: warehouse publish без CRM tables; на GCE publish с Mac уходит |
| Путают shadow LLM и решение методиста | явные labels; training_use отдельно |
| Слишком рано BigQuery | не в DoD v1 |
| 8 GB мало при пике scoring+RAG | staging smoke; запасной `e2-standard-4` (16 GB) |
| VM ops (патчи, бэкапы, TLS) | snapshot disk daily; Caddy/nginx + Let's Encrypt; OS updates |
| Cutover DNS / домен | подробно §11.8-11.10 |

---

## 9. Definition of Done v1 (A + минимальный B) - Render only

1. В очереди «Вчера» и таблице случаев видны visit_id, дата, ФИО, спец., филиал; patient_id для методиста.
2. Разбор открывается на весь экран.
3. Решение методиста содержит вердикты по 3 вопросам + развёрнутый текст.
4. Save пишет review pack (МО + system/LLM snapshot + decision), список версий доступен в UI.
5. BigQuery не обязателен; путь экспорта описан в фазе C.
6. Хост v1: **остаёмся на Render** (persistent disk + SQLite CRM/review packs). GCE cutover (§11) - отдельно, не в DoD этого PR.

---

## 10. Предлагаемый порядок работ

1. Согласовано: **SQLite primary** (да), patient_id в UI методиста (да), **GCE отложен** - сначала всё на Render.
2. Реализовать A1-A3 + B1-B3 в одном PR на Render.
3. H1-H4 (Finland VM) - только после явного go от владельца.
4. C - после накопления разметок; BQ только как analytics sink.

---

## 11. Хостинг: GCP VM Instances (`europe-north1`, Finland)

Связь с `2026-08-04-mo-runtime-stabilization-v1.md`: там GCP описан как Cloud Run Job + GCS (фаза C). Здесь рассматриваем **более простой и часто дешевле вариант для нашего объёма** - одна (или две) Compute Engine VM в EU.

### 11.1 Зачем VM, а не Render / не сразу BigQuery

| Критерий | Render (сейчас) | GCE VM europe-north1 | BigQuery |
|--|--|--|--|
| Стоимость always-on + диск | выше при росте disk/фоновых job | предсказуемый e2 + pd-ssd | дорого как OLTP, ок как analytics |
| Латентность до EU/Минск | US Oregon (сейчас) | EU Finland - ближе к пользователю/MIS VPN-пути | n/a для UI |
| Persistent data | Render disk + хрупкий SSH publish | локальный PD, worker пишет сам | не для CRM write |
| Review packs / CRM | ок, но publish-контур сложный | один диск = web+CRM+judges+packs | export only |
| Ops | меньше | больше (OS, TLS, backup) | managed SQL analytics |

Вывод: для пакетов разбора и daily МО **GCE VM в Finland** - разумный целевой хост; BigQuery - не замена VM.

### 11.2 Рекомендуемый размер (8 GB)

Интерпретация запроса «8 мегабайт памяти» → **8 GB RAM**.

Стартовый тип:

| Параметр | Значение |
|--|--|
| Region | `europe-north1` (Finland) |
| Machine | `e2-standard-2` (2 vCPU, **8 GB** RAM) |
| Boot + data | pd-balanced или pd-ssd **50-100 GB** (warehouse, secure_cases, judges, review_packs) |
| OS | Debian 12 / Ubuntu LTS |
| Оценка compute (order of magnitude) | on-demand ~$0.074/h ≈ **~$50-55/мес**; 1y CUD ≈ **~$30-35/мес** (без диска/egress; сверить в калькуляторе GCP перед закупкой) |
| Диск 100 GB pd-balanced | порядка единиц-десятков $/мес (уточнить в калькуляторе) |

Когда 8 GB мало:

- одновременный heavy scoring (pandas) + uvicorn + Gemini batch;
- тогда: `e2-standard-4` (16 GB) **или** вторая VM/spot только под nightly worker (start 05:50 Minsk → stop после publish).

Для текущего режима `startup_mode=manifest` + SQLite BI **8 GB обычно достаточно** для web; worker лучше не пиковать вместе с пиковым трафиком или вынести на отдельный schedule.

### 11.3 Целевая схема на VM

```text
                    GitHub origin/main
                           |
                    CI build image / rsync release
                           |
              +------------+-------------+
              | GCE europe-north1        |
              |  e2-standard-2 (8 GB)    |
              |  + persistent disk       |
              |                          |
              |  nginx/Caddy → uvicorn   |  ← HTTPS, methodist UI/API
              |  cron 06:00 Europe/Minsk |  ← MO daily + LLM action judge
              |  /data/medical_exams/    |  ← warehouse, CRM, judges, review_packs
              +--------------------------+
                           |
              (опционально) GCS backup nightly
                           |
              (опционально) BigQuery load gold packs
```

Правила:

1. Web и данные на **одном PD** - нет SSH-publish с Mac как primary.
2. CRM / `crm_review_pack` никогда не затираются «тонким» publish.
3. Секреты: Secret Manager или env-файл вне git (как сейчас `.env`).
4. Deploy: только SHA = `origin/main` (тот же guard, что Render release).
5. Backup: daily snapshot диска + опционально `sqlite3 .backup` перед scoring.

### 11.4 Варианты cutover

| Вариант | Плюсы | Минусы | Когда |
|--|--|--|--|
| **H-all**: web+worker+data на одной VM Finland | дешевле всего, простой диск | одна точка отказа; нужен ops | рекомендуемый default |
| **H-split**: web Render, worker+data GCE | меньше риска для UI | снова sync/publish | если DNS/TLS на VM тормозит |
| **H-run**: Cloud Run web + GCE/Job worker | managed HTTPS | сложнее shared SQLite | если отказываемся от SQLite на web |

Для review packs предпочтителен **H-all**: пакеты пишутся локально в ту же SQLite, что читает UI.

### 11.5 Сравнение с Cloud Run (из runtime-плана)

- Cloud Run удобен для stateless web, но **SQLite + CRM + review packs** требуют volume (Filestore/GCS fuse) - дороже и хрупче, чем PD на VM.
- Cloud Run Job хорош для daily ETL; VM проще, пока объём ~сотни-тысячи МО/день.
- Решение: **сначала GCE VM 8 GB в europe-north1**; Cloud Run оставить запасным путём, если ops VM не понравится.

### 11.6 Поэтапный переход Render → GCE (календарь)

Сейчас прод: сервис Render `protocol`, URL **`https://protocol-bimy.onrender.com`**, диск `/var/data/medical_exams`, деплой только с `origin/main`.

#### Фаза H0 - подготовка (T-7 … T-3 дня)

1. GCP project + billing + API (Compute, optionally IAP, Secret Manager).
2. Создать VPC / firewall:
   - inbound `tcp/80`, `tcp/443` from `0.0.0.0/0`;
   - SSH только через **IAP** или allowlist IP (не 0.0.0.0/22 открытым).
3. Создать VM `e2-standard-2` в `europe-north1-a` (или `-b`) + persistent disk 50-100 GB.
4. Установить Docker или native Python venv + Caddy/nginx; mount PD → `/data/medical_exams`.
5. Завести **staging URL** сразу (не ждать cutover):
   - либо временный `https://PROTOCOL_VM_IP` (только для команды);
   - либо поддомен `staging.ваш-домен` (см. §11.8).
6. Один раз скопировать данные с Render (rsync по SSH на Render → локально/GCS → VM):
   - `warehouse/` (включая CRM),
   - `reports/`, `secure_cases/`, `state/`, `llm_action_judge/`, `public/`.
7. Поставить секреты (`.env` / Secret Manager): `METHODIST_TOKEN`, `GOOGLE_API_KEY`, DB password MIS.
8. Smoke на staging: `/health`, `/api/version`, login методиста, case detail, daily «Вчера».

Критерий выхода H0: staging на Finland отвечает тем же `BUILD_VERSION`, что `main`; warehouse не пустой; CRM statuses на месте.

#### Фаза H1 - параллельный прогон (T-2 … T-0)

1. Включить на VM **read-only** или shadow cron (без записи в MIS): nightly ETL → диск VM.
2. Сверить метрики «Вчера» Render vs VM (coverage, action queue count, overall avg).
3. Прогнать LLM action-judge batch на VM (geo Finland обычно без block).
4. Настроить CI deploy на VM (rsync/ssh или pull `main` + restart systemd) **без** отключения Render.
5. Зафиксировать runbook rollback (§11.9).

Критерий выхода H1: 1-2 дня подряд цифры VM ≈ Render (допуск зафиксировать, напр. ±1 кейс в очереди).

#### Фаза H2 - cutover домена / URL (окно 15-60 мин)

См. §11.8 подробно. Кратко:

1. Freeze: не писать CRM на Render за 15 мин до переключения (или принять, что последние клики могут потеряться - лучше объявить окно).
2. Финальный rsync CRM/warehouse Render → VM (только дельта).
3. Переключить DNS / URL на VM.
4. Проверить TLS, `/api/version`, сохранение тестового review pack.
5. Render оставить **cold standby** (сервис не удалять 7-14 дней).

#### Фаза H3 - стабилизация (T+1 … T+14)

1. Mac launchd → `fallback-only` (как в runtime-плане).
2. Ежедневный snapshot PD + тест restore на 3-й день.
3. Мониторинг: uptime, disk %, latency `/health`, Telegram на fail cron.
4. Через 7-14 дней: остановить Render billing / уменьшить план, архивировать disk snapshot.

### 11.7 Чеклист миграции на VM (H)

- [ ] GCP project + billing; регион `europe-north1`.
- [ ] VPC, firewall: 80/443 world; SSH только IAP или allowlist.
- [ ] VM `e2-standard-2` + PD 50-100 GB, mount `/data/medical_exams`.
- [ ] Staging URL + копия данных с Render.
- [ ] systemd: `protocol-web`, cron `mo-daily`, logrotate, Caddy TLS.
- [ ] Параллельный прогон 1-2 дня (сверка «Вчера»).
- [ ] Cutover DNS/домена (§11.8).
- [ ] Smoke prod: health, version, case detail, review-pack save.
- [ ] Render cold standby 7-14 дней; Mac fallback-only.
- [ ] Snapshot schedule + тест restore.

### 11.8 Как переносится домен / URL

Важно: **`protocol-bimy.onrender.com` нельзя «перенести» на GCP.**  
Это hostname Render. Снаружи возможны три стратегии.

#### Вариант D1 - свой домен (рекомендуется)

Нужен DNS, которым мы управляем (Cloudflare / registrar), например:

- `app.protocol.by` или `mo.example.com` (имя согласовать).

Шаги:

1. **До cutover:** на VM поднять Caddy/nginx с TLS (Let's Encrypt) на этот hostname; DNS ещё **не** на VM - проверить через `/etc/hosts` или Cloudflare orange-cloud off + временный A на staging.
2. В Cloudflare (или аналог) создать запись:
   - `A` → статический **внешний IP** VM (зарезервировать GCP External IP, чтобы не сменился после stop/start);
   - или `CNAME` на DNS-имя VM (хуже: IP GCE лучше держать reserved).
3. TTL перед окном: снизить до **60-300 сек** за сутки.
4. В окно cutover: переключить `A` с (если был) старого целевого / с «заглушки» на IP Finland.
5. Проверить: `dig +short app…` → IP VM; `curl -fsS https://app…/api/version`.
6. Обновить все клиенты/скрипты:
   - `PROTOCOL_PROD_URL`, launchd, CI `prod-url`, закладки методистов, Telegram alerts.
7. Опционально: на Render оставить redirect/maintenance page «переехали на …» если кто-то зайдёт на старый onrender URL.

Плюсы: нормальный бренд, независимы от Render.  
Минусы: нужно купить/иметь домен и SSL (Let's Encrypt решает).

#### Вариант D2 - временно только IP / новый hostname GCP

- Пользователи ходят на `https://X.X.X.X` или `https://protocol-finland.example-nip.io` - **не для продакшена методистам**.
- Только для внутренней приёмки H0/H1.

#### Вариант D3 - оставить `*.onrender.com` как витрину + reverse proxy

Теоретически: Render web = тонкий proxy на GCE.  
**Не рекомендуем:** двойная оплата, сложнее TLS/таймауты, смысл переезда теряется.

#### Что делать с текущим `protocol-bimy.onrender.com`

| Действие | Когда |
|--|--|
| Не удалять сервис | 7-14 дней после cutover |
| Поставить static maintenance / redirect на новый домен | сразу после D1 cutover |
| Обновить CI: `Production Render release` → `Production GCE release` (SSH/systemd) | в том же PR, что cutover |
| Закрыть Render disk billing | после подтверждения snapshot на GCP |

#### DNS-схема cutover (D1)

```text
До:
  браузер → protocol-bimy.onrender.com → Render Oregon → /var/data/...

После:
  браузер → app.ваш-домен
                 │
                 │  A-record (TTL low)
                 ▼
           GCP External IP (reserved)
                 │
                 ▼
           Caddy :443 → uvicorn :8000
                 │
                 ▼
           /data/medical_exams (PD Finland)
```

Окно простоя: при TTL 60s обычно **2-15 минут** рассинхрона DNS у части клиентов.  
Митигация: объявить окно; держать Render online; при проблеме откатить A-record назад (<5 мин).

### 11.9 Rollback

1. DNS A-record снова на Render (или снять custom domain с VM) - UI снова Oregon.
2. Если за окно на VM успели писать CRM/review packs - rsync **дельта обратно** на Render disk до отката DNS (или смириться с потерей кликов за окно - лучше freeze).
3. Вернуть CI на Render release.
4. Разбор инцидента; повтор cutover не раньше чем через 24ч.

### 11.10 Риски перехода (сводка)

| Риск | Вероятность | Влияние | Митигация |
|--|--|--|--|
| Пользователи продолжают открывать `*.onrender.com` | высокая | путаница, старые данные | redirect/maintenance; рассылка; обновить закладки |
| Потеря CRM за окно cutover | средняя | решения методиста | freeze записи + финальный rsync; rollback-rsync |
| Долгий DNS TTL у резолверов | средняя | часть юзеров на старом хосте часы | снизить TTL заранее; держать Render online |
| Let's Encrypt fail (80 закрыт / rate limit) | низкая | нет HTTPS | открыть 80; staging cert заранее; запасной cert |
| 8 GB OOM на scoring | средняя | падение web ночью | worker в off-peak; лимит concurrency; запас `e2-standard-4` |
| Утечка ПДн при rsync/snapshot | средняя | compliance | шифрование диска GCP, SSH only IAP, не копировать в git/Telegram |
| Сломан CI deploy на VM | средняя | застревание версий | health-gate как сейчас; ручной rollback image |
| MIS недоступна с Finland (firewall) | средняя | daily ETL fail | проверить whitelist IP VM у хостинга МИС; VPN-путь |
| Gemini/API geo | низкая из EU | batch fail | Finland обычно ok; иметь fallback ключ/регион |
| Один VM = SPOF | высокая (архитектура) | downtime | snapshot + быстрый recreate; позже MIG/вторая зона |
| Стоимость egress EU→мир | низкая при нашем трафике | сюрприз в счёте | мониторинг Billing budget alert |
| Забыли обновить скрипты `PROTOCOL_PROD_URL` | высокая | «прод» смотрят не туда | чеклист cutover + один source of truth в docs |

### 11.11 Что не меняется в продукте

- Контракт review pack (§4) и фазы A/B UI - те же.
- Primary overall warehouse score по-прежнему не заменяется решением методиста.
- Gemini ключи остаются; geo с Finland для API обычно ок (в отличие от BY geo-block).
- Ветка деплоя по-прежнему только merged `origin/main`.
