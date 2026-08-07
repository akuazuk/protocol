# Разделение контуров: GCP сейчас, МИС-мост с Mac, BY позже (v1)

Дата: 2026-08-07  
Статус: active  
Уточнение владельца (2026-08-07 вечер):  
**сначала почти всё на GCP**; подключение к БД МИС пока с **Mac**, потом с **GCP**,  
потом канон переезжает на **BY-сервер**; GCP в конце остаётся для платных LLM.  
Связанные:

- `2026-08-04-mo-runtime-stabilization-v1.md` - Docker; фаза C уточнена здесь
- `2026-08-07-mo-auto-llm-on-disk-v1.md`
- `docs/deploy/persistent_disk.md`, `docs/mo-daily-pipeline.md`
- `.cursor/rules/mis-mariadb.mdc`, `gemini-via-render.mdc`

---

## 1. Дорожная карта владельца (источник истины)

Три эпохи - проектируем код/Docker **сразу** под все три, иначе потом будет больно резать.

| Эпоха | Web + warehouse + файлы МО | Платный LLM | Подключение к MariaDB МИС |
|--|--|--|--|
| **E0 сейчас** | Render + Mac launchd | Render (не Mac) | Mac (VPN off) |
| **E1 цель ближайшая** | **GCP** (GCE или Cloud Run + диск) | **GCP** | **Mac-мост** (ETL на Mac → артефакты на GCP) |
| **E2** | GCP | GCP | **GCP** (VPN/tunnel/allowlist с cloud) |
| **E3 цель дальняя** | **BY-сервер** (канон кода и БД) | **GCP only** | **BY** (VPN/LAN к Kravira) |

Правила на все эпохи:

1. Канон warehouse/CRM/secure_cases в каждый момент живёт **в одном** месте (leader), не в двух.
2. LLM jobs - отдельный образ `gcp-llm`; не смешивать с ETL МИС в одном контейнере без нужды.
3. Клиент МИС (`mis_bridge`) - **отдельный модуль/entrypoint**, который можно запускать на Mac → GCP → BY, меняя только host/env.
4. Пароль МИС не обязан жить в GCP в эпоху E1 (остаётся на Mac); в E2 - Secret Manager GCP; в E3 - только BY.
5. Docker-образы одни и те же во все эпохи; меняется оркестратор и где крутится `mis_bridge`.

---

## 2. Контекст (как сейчас = E0)

```text
Mac launchd --ETL МИС + score--> publish SSH --> Render disk + web
                                              └─ LLM grade на Render
```

Боли: Mac writer, SSH, Oregon, смешение пакетов.  
Уже есть: night LLM на Render, trigger после publish (#37), action-judge full queue (#38).

---

## 3. Архитектура по эпохам

### E1 - всё на GCP, МИС пока с Mac (делать первым после границ репо)

```text
Mac (тонкий mis_bridge / launchd)
  VPN off → MariaDB Kravira
  пишет day CSV / raw extract
       |
       v  upload (GCS или API BY-soon / GCP ingest)
GCP (primary)
  - api web
  - mo-pipeline: score, warehouse, recompute, CRM
  - volume/PD или GCS+sqlite strategy
  - gcp-llm jobs: Gemini grade + action-judge
```

Mac **не** считает primary score/warehouse (по возможности): только extract+upload.  
Если временно score ещё на Mac - явно пометить transitional и выключить ASAP.

### E2 - МИС с GCP

Тот же GCP stack; `mis_bridge` переезжает в Cloud Run Job / sidecar на GCE  
с сетевым доступом к `178.163.240.131:6330` (VPN appliance / allowlist / tunnel).  
Mac launchd - fallback-only.

### E3 - BY = home of truth, GCP = LLM farm

```text
BY: api + mo-pipeline + warehouse + mis_bridge + файлы
GCP: только gcp-llm (grades/judges) ↔ inbox/outbox contract
```

Это конечная схема из исходного решения «код и базы в РБ».

---

## 4. Метрики

| Метрика | E0 | Цель E1 | Цель E3 |
|--|--|--|--|
| Web primary | Render | GCP | BY |
| Warehouse leader | Render disk | GCP | BY |
| LLM | Render | GCP | GCP |
| MIS connect host | Mac | Mac bridge | BY |
| iMac нужен ночью | да | только extract | нет |
| Docker images | нет | by-app + gcp-llm (+ mis_bridge) | те же |

---

## 5. Организация файлов сейчас (чтобы E1→E2→E3 было дёшево)

### 5.1 Каталоги

```text
services/
  api/                 # web (rag_server обёртка)
  mo_pipeline/         # score, recompute, warehouse upsert (без Gemini SDK)
  mis_bridge/          # ТОЛЬКО выгрузка из MariaDB → day artifacts
                       # entrypoint один; env: MIS_DSN, OUT_DIR, RUN_HOST=mac|gcp|by
  llm_worker/          # GCP grade/judge CLI
deploy/
  gcp-app/             # Dockerfile + compose/terraform для эпохи E1/E2 (web+pipeline)
  gcp-llm/             # Dockerfile LLM job + job-contract.md
  by-home/             # Dockerfile/compose для эпохи E3 (пока заготовка)
  mac-bridge/          # launchd/plist + скрипт upload extract → GCS
packages/              # optional later shared libs
```

Имена образов:

| Image | Эпохи | Entrypoint |
|--|--|--|
| `protocol-gcp-app` | E1, E2 (E3 optional read-replica) | api + mo_pipeline |
| `protocol-gcp-llm` | E1-E3 | grade_day / judge |
| `protocol-mis-bridge` | E1 Mac, E2 GCP, E3 BY | extract_day |
| `protocol-by-home` | E3 | api + pipeline (+ bridge) |

Не делать один «бог-образ» на всё - иначе Mac-мост потащит Gemini и наоборот.

### 5.2 Контракт данных между Mac-мостом и GCP (E1)

`deploy/mac-bridge/extract-contract.md` + `deploy/gcp-llm/job-contract.md`:

**Mac → GCP (extract):**

- `mo_YYYY-MM-DD.csv` (или parquet) + meta (`extracted_at`, row_count, checksum)
- **без** password; upload через SA key / signed URL

**GCP internal:**

- score → `secure_cases` + queue
- submit llm job → grades/judges → recompute

**GCP → BY (будущее E3):** миграция volume/snapshot + DNS; контракт LLM тот же inbox/outbox.

### 5.3 Paths на диске GCP (E1)

```text
$MO_DATA_ROOT/
  inbound/extract/     # то, что прислал Mac-мост
  warehouse/
  secure_cases/
  reports/ state/
  llm_outbox/ llm_inbox/
  gold_review/
```

### 5.4 Зависимости

- `requirements-rag.txt` → api  
- `requirements-llm-worker.txt` → gcp-llm (минимальный Gemini stack)  
- `requirements-mis-bridge.txt` → pymysql/sqlalchemy (+ pandas если нужен extract)  
- pipeline deps отдельно; **не** ставить Gemini в mis_bridge образ

### 5.5 Импорты

- `mis_bridge` не импортирует `rag_server`, Gemini, frontend  
- `llm_worker` не импортирует MIS DSN / sql_epam  
- `mo_pipeline` может читать inbound extract, не ходит в Gemini напрямую

---

## 6. Docker: нужен ли на Mac сейчас?

| Вопрос | Ответ |
|--|--|
| Установлен ли Docker на этом Mac? | **Нет** (на 2026-08-07) |
| Нужен ли для эпохи E1 на проде? | На Mac для **моста МИС** - Docker **не обязателен**: достаточно Python launchd + upload в GCS |
| Нужен ли для разработки границ/образов? | **Желательно да** - локально `docker build` api/llm/bridge; иначе сборка только в GitHub Actions |
| Что поставить, если ставить | Docker Desktop for Mac **или** Colima (легче по ресурсам). Не OrbStack-обязательно |
| Когда точно понадобится локально | Отладка compose `gcp-app` + volume; воспроизведение CI fail |

**Практическая рекомендация:**  
- Если начинаем фазу A на этой неделе и хочешь проверять образы у себя - **скачай Docker Desktop (или Colima)**.  
- Если ок опираться на CI - можно **не ставить сейчас**; Mac-мост МИС продолжит жить на системном Python как launchd.  
- Для переноса на GCP сервер Docker нужен **там**, не обязательно на ноутбуке.

---

## 7. Фазы реализации (с учётом E1→E3)

### Фаза A - границы в репо (без железа) - 3-7 дней

- [x] A1. `services/{api,mo_pipeline,mis_bridge,llm_worker}/README.md` (владение файлами).
- [x] A2. Контракты: `deploy/mac-bridge/extract-contract.md`, `deploy/gcp-llm/job-contract.md` + fixtures.
- [x] A3. `requirements-llm-worker.txt`, `requirements-mis-bridge.txt`.
- [x] A4. `Dockerfile` ×3: `gcp-app`, `gcp-llm`, `mis-bridge` (+ заготовка `by-home`).
- [x] A5. CI: build трёх образов на PR (без PHI) + stub `by-home`.
- [x] A6. Вынести extract МИС в явный CLI `mis_bridge` (обёртка над текущим daily extract), env `RUN_HOST=mac`.
- [x] A7. Paths `inbound/extract`, `llm_inbox`/`outbox` в коде (env defaults).
- [x] A8. Индекс plans: runtime C → этот план.

Критерий A: образы собираются; контракты описаны; поведение E0 не сломано.

### Фаза B - E1 cutover на GCP (приложение), Mac = только МИС-мост - 1-3 недели

- [x] B1. GCP project + GCE (проще для SQLite disk) или Cloud Run+Filestore; регион с рабочим Gemini.
  - project `protocol-home-e1`, VM `protocol-app` e2-standard-2, PD 50GB, zone `europe-central2-a`, IP `34.118.21.47`, bucket `gs://protocol-home-e1-inbound` (см. `deploy/gcp-app/INVENTORY.md`).
- [~] B2. Перенос Render `medical_exams` → GCP PD/GCS; web на GCP; `/api/version` smoke.
  - web staging live: `http://34.118.21.47:8000` (`deploy_to_gce.sh`); `/api/version` ok.
  - migrate warehouse/secure_cases с Render - ещё нет.
- [ ] B3. LLM jobs на том же GCP (`gcp-llm`); убрать зависимость night grade от Render SSH.
- [ ] B4. Mac launchd: режим `extract-upload-only` → GCS `inbound/extract`; score/recompute на GCP.
- [ ] B5. DNS/домен с Render на GCP (или временный hostname).
- [ ] B6. Rollback: Render snapshot + старый publish path.

Критерий B: «Вчера» с GCP без Render disk; Mac ночью только extract.

### Фаза C - E2 МИС с GCP

- [ ] C1. Сеть до Kravira с GCP (VPN/allowlist) - отдельное согласование с доступом к МИС.
- [ ] C2. `mis_bridge` Job на GCP; Secret Manager для DB password.
- [ ] C3. Mac bridge fallback-only.

### Фаза D - E3 BY home of truth

- [ ] D1. BY сервер Docker; migrate volume + warehouse.
- [ ] D2. Web+pipeline+mis_bridge на BY; GCP только `gcp-llm` + transit bucket.
- [ ] D3. Выключить Mac primary полностью.

---

## 8. Лишний мусор в репо при переносе (учтен)

Раньше в этом плане **не было** явного блока - добавляется здесь.  
Связанный план гигиены: `2026-08-04-repo-sections-archive-v2.md` (карта разделов, archive/).

### 8.1 Что не тащить в образы и не раздувать git при E1

| Класс | Примеры | Политика |
|--|--|--|
| Локальные PDF/дампы | `minzdrav_protocols/**/*.pdf`, сырой `output/`, `corpus_vector_index/` | уже/должно быть в `.gitignore`; **не** `COPY` в Docker; не коммитить «заодно» |
| ML batch dumps | `ml/experiments/batch_*`, `data/ml/reports/` | `archive/ml-experiments` или ignore; не в `gcp-app`/`gcp-llm` |
| Конкурс / leftovers | `archive/docs/konkurs`, root orphan assets | не трогать в PR миграции |
| Секреты | `.env`, `KRAVIRA_*`, Gemini keys | никогда в образ через COPY; только Secret Manager / env runtime |
| PHI на диске | `secure_cases`, `gold_review`, grades | только volume/GCS; не в git, не в CI artifact |
| Исторические scripts | плоский `scripts/*.py` ~180 | **не** массовый `git mv` в одном PR; wrappers/README ownership сначала |

### 8.2 Правила рефакторинга структуры

1. **Малые PR**, один слой за раз: (a) README ownership → (b) Dockerfiles → (c) thin CLI wrappers → (d) optional physical move.
2. **Запрет** «переложить весь monorepo за выходные» - ломает параллельных агентов и blame.
3. Новые файлы миграции только под `services/`, `deploy/{gcp-app,gcp-llm,by-home,mac-bridge}/`.
4. Перед большим PR: `scripts/ops/smoke_repo_layout.sh` + список путей в описании PR «что не копируется в образ».
5. Локальный мусор на checkout (untracked PDF и т.п.) **не** попадает в `git add -A`; только явные pathspecs.

### 8.3 Docker COPY allowlist (ориентир)

`gcp-llm`: `clinical_knowledge/llm*` + grade/judge scripts + `requirements-llm-worker.txt` + минимальные deps.  
`mis-bridge`: extract scripts + pymysql stack.  
`gcp-app`: api + mo_pipeline + frontend static + rag requirements - **без** PDF corpus и без ML dumps.

---

## 9. Несколько компьютеров и агентов (учтен)

Базовый канон уже есть (обязателен и для этой миграции):

- `AGENTS.md`
- `docs/deploy/multi-agent-single-repo-render-runbook-v2.md`
- `docs/deploy/two-computers-daily-checklist.md`
- `.cursor/rules/repository-coordination.mdc`

Для **этого** плана дополнительно:

| Правило | Зачем |
|--|--|
| Одна task-ветка на одного агента/PC (`cursor/by-gcp-llm-split-…-pcN`) | Нет shared dirty branch |
| Draft PR сразу: список каталогов `services/`, `deploy/…` | Другие агенты видят ownership |
| Не параллелить два PR, оба трогающие `rag_server.py` / `publish_mo_*` / Dockerfiles без очереди | Конфликты cutover |
| Release/deploy только после squash в `origin/main` + `render_release` / будущий GCP deploy guard | Как сейчас с Render |
| Handoff при смене PC: новый файл в `docs/handoff/` + версия в проде | Mac sleep / другой агент |
| Данные МО не синхронизировать через git между компьютерами | Только Render/GCP disk / GCS |
| `BUILD_VERSION` UTC stamp в каждом осмысленном PR | Нет коллизий rN между PC |

Anti-conflict матрица (кто может трогать параллельно):

| Зона | Владелец плана split | Можно параллельно другому агенту |
|--|--|--|
| `deploy/**`, `Dockerfile*`, `services/**` | этот план | нет без согласования |
| `frontend/web/shared/mo-app.js` продуктовые фичи | другие планы | да, если не docker/compose |
| scorer / suggest / ICD directory | другие планы | да |
| `scripts/run_mo_daily_launchd.sh`, publish | согласовать с runtime + этим планом | осторожно |

Ветка (пример):

```bash
scripts/ops/git_task_start.sh by-gcp-llm-split --pc=pc1 \
  --branch=cursor/by-gcp-llm-split-v1-pc1
```

---

## 10. Риски

| Риск | Митигация |
|--|--|
| Два writer (Mac score + GCP score) | В E1 явно один leader; Mac только extract |
| PHI в GCS transit | TTL, bucket IAM, no public ACL |
| Gemini geo с GCP | выбрать регион/проект где API стабилен (проверить до cutover) |
| VPN МИС с GCP (E2) | не блокировать E1; мост Mac - осознанный промежуточный шаг |
| SQLite на Cloud Run | в E1 предпочтительна **GCE + PD** |
| Docker не на Mac | CI builds; Mac bridge без Docker |
| Закоммитили GB мусора при «организации» | §8 allowlist + pathspec add; ревью PR на `git diff --stat` |
| Два агента ломают cutover | §9 draft PR + матрица зон |
| Гигантский rename PR | дробить; shims вместо big-bang mv |

---

## 11. Владение файлами (кратко)

План владеет: `services/**`, `deploy/{gcp-app,gcp-llm,by-home,mac-bridge}/**`,  
`Dockerfile*`, `requirements-llm-worker.txt`, `requirements-mis-bridge.txt`.

Не смешивать в одном PR с продуктовым suggest/МКБ без нужды.

---

## 12. Definition of Done v1 (документация + готовый каркас)

1. Дорожная карта E1→E3 зафиксирована (этот файл).
2. Три+ образа и контракты extract/LLM описаны.
3. `mis_bridge` как отдельный entrypoint (можно пока thin wrapper).
4. Понятно: Docker на Mac опционален для моста; обязателен на GCP (и позже BY).
5. Явно учтены: hygiene (§8) и multi-agent/multi-PC (§9).

---

## 13. Вне скоупа

- Покупка BY железа до завершения E1 (можно выбирать параллельно, не блочит A/B).
- Перенос самой MariaDB Kravira.
- Смена scorer/BI.
- Полная физическая уборка всего плоского `scripts/` (отдельные PR по archive-плану).

---

## 14. Статус

План обновлён: E1 GCP-first + МИС с Mac; **добавлены §8 (мусор/гигиена) и §9 (агенты/PC)**.  
**Фаза A закрыта в репо:** services READMEs, контракты+fixtures, split requirements,  
Dockerfiles (`gcp-app` / `gcp-llm` / `mis-bridge` / `by-home` stub), CI `docker-images`,  
thin CLI `services.mis_bridge` / `services.llm_worker`.

### Решения владельца (фаза B, 2026-08-07)

| Вопрос | Решение |
|--|--|
| GCP project | **создан** `protocol-home-e1` (аккаунт `aicoursesus@gmail.com`; не reuse `gen-lang-client-*`) |
| Billing | linked `01D5C2-ECFF77-88FFEC` (`My Billing Account`) |
| Хост | **GCE + persistent disk** (не Cloud Run) |
| Регион | **EU ближе к Минску** → канон: `europe-central2` (Warsaw); запасной `europe-north1` (Finland, уже в methodist-плане) |
| Домен | сначала **временный hostname** GCP (IP / `*.nip.io` / Cloud DNS temp); Render DNS не трогать |
| Gemini | сейчас AI Studio key (`google-generativeai`) - регион VM важен для latency/ops; smoke LLM с GCE до cutover |
| APIs | compute, storage, secretmanager, artifactregistry, iam, cloudresourcemanager |

**B1 done:** VM `protocol-app` RUNNING, `/var/data` mounted, Docker installed, GCS inbound bucket.
**B2 partial:** web staging live `http://34.118.21.47:8000` (`deploy_to_gce.sh`); env via
`.env.gcp-staging` (см. `deploy/gcp-app/ENV-MIGRATION.md`). Warehouse с Render ещё не мигрирован.
Инвентарь: `deploy/gcp-app/INVENTORY.md`.
Следующий шаг: migrate `medical_exams` Render → PD; Secret Manager для ключей; HTTPS temp host.
