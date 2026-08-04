# Стабилизация runtime МО: Docker, вынос пайплайна с Mac, опционально GCP (v1)

Дата: 2026-08-04  
Статус: active  
Автор: агент + владелец продукта  
Связанные планы:

- `2026-07-28-mo-daily-bi-platform-v1.md` - ежедневный приём и витрина;
- `2026-07-30-mo-analytics-bi-redesign-v1.md` - BI UI и доставка данных в прод;
- `2026-08-03-ci-release-concurrency-v3.md` - PR-only `main` и GitHub Action release (completed);
- `AGENTS.md` + `docs/deploy/multi-agent-single-repo-render-runbook-v2.md` - координация агентов.

---

## 1. Контекст и диагноз

Продукт состоит из двух разных систем, которые сейчас связаны хрупко:

| Система | Где крутится | Задача |
|---|---|---|
| Web API / UI | Render `protocol` (`protocol-bimy.onrender.com`) | читать витрину, отдавать МО Аналитику |
| Daily MO pipeline | Mac launchd (`by.protocol.mo-daily*`) | ETL из MariaDB → scoring → warehouse → publish |

Факты на 2026-08-04:

- код в проде актуален: `c244c98` / `2026-08-03-r24-ci-release-handoff`;
- данные за `2026-08-03` **есть** локально и на Render disk;
- день остаётся `partial` из-за `llm_queue_pending` (80 визитов), не из-за отсутствия ETL;
- `publish_mo_to_render.py` часто шумит: SSH hostkey signature errors, freshness check 403 без `METHODIST_TOKEN` в launchd;
- в репозитории **нет Dockerfile**;
- `render.yaml` **не управляет** живым продом (там приостановленный `protocol-rag`).

Вывод: переезд веб-части на Google Cloud **сам по себе** не стабилизирует «Вчера». Нужно сначала отвязать пайплайн от ноутбука и сделать publish без SSH с Mac.

---

## 2. Цель

К концу плана:

1. «Вчера» в проде к **07:00 Europe/Minsk**, `lag_days <= 1`.
2. День либо `success`, либо явный `partial` с понятной причиной в UI/health (не «тишина»).
3. Primary pipeline **не зависит** от включённого iMac.
4. Один Docker-образ (web) + один job-образ (worker) воспроизводят прод локально и в CI.
5. Деплой кода остаётся только через merged `origin/main` (уже сделано).
6. GCP - **опция фазы C**, не обязательный первый шаг.

Не цель этого плана: переписать scorer v4, BI-редизайн экранов, перенос MariaDB MIS.

---

## 3. Что уже в проде (база)

- GitHub branch protection + PR-required `main`.
- Action `Production Render release` с `concurrency: production-render`.
- `scripts/ops/render_release_main.sh` принимает только SHA = HEAD `origin/main`.
- Старые `render_promote_main.sh` / `deploy_promote_main_after_push.sh` fail-closed.
- Persistent disk `/var/data/medical_exams` на Render читается API.
- Локальный launchd всё ещё primary writer.

---

## 4. Метрики: было / цель

| Метрика | Было (04.08) | Цель фазы A | Цель фазы B | Цель фазы C (если GCP) |
|---|---|---|---|---|
| Лаг «Вчера» в проде | данные есть, статус `partial` | явный статус + алерт | `success` или осознанный `partial`, lag <= 1 | lag <= 1 без Mac |
| Primary host пайплайна | iMac launchd | iMac, но publish/token починены | cloud worker | Cloud Run Job / GCE |
| Publish path | SSH+rsync с Mac | тот же, но идемпотентный + алерт | worker пишет в общий storage | GCS + mount/sync |
| Freshness verify после publish | часто 403 | 200 + token в env | 200 | 200 |
| Docker | нет | draft Dockerfile web | web+worker в CI | Artifact Registry |
| Хостинг web | Render | Render | Render (default) | Cloud Run optional |
| Зависимость от домашнего Mac | 100% | 100% | < 20% (fallback only) | 0% primary |

---

## 5. Целевая архитектура

```text
                    GitHub origin/main
                           |
              +------------+------------+
              |                         |
     build web image            build worker image
              |                         |
     Render / Cloud Run           Scheduler 06:00 Minsk
     (API + UI only)                    |
                                  MO Worker Job
                                  1) ETL MariaDB
                                  2) score + LLM queue drain
                                  3) upsert warehouse
                                  4) write artifacts
                           |
              shared data plane
              - warehouse sqlite (или позже SQL)
              - reports / state / public / secure_cases
              Render disk  ИЛИ  GCS (+ sync на disk)
```

Правила разделения:

1. **Web никогда не считает дневной МО.**
2. **Worker никогда не принимает пользовательский HTTP** (кроме health для job).
3. Publish не затирает CRM методистов в проде (как сейчас в `publish_mo_to_render.py`).
4. Секреты только в Secret Manager / Render env / launchd env, не в git.

---

## 6. Фазы и шаги

### Фаза A - стабилизация без смены хостинга (3-5 дней)

Цель: «Вчера» предсказуемо, алерты честные, Mac ещё primary.

- [ ] A1. Закрыть/дожать `llm_queue_pending` за 2026-08-01..03 (или явная политика: день `success` при coverage>=99% и очередь advisory).
- [ ] A2. В launchd plist / env добавить `METHODIST_TOKEN` для freshness-check после publish.
- [ ] A3. Починить/обернуть publish: ретраи SSH, понятный exit code, Telegram при fail, не считать 403 «тихим успехом».
- [ ] A4. Снять stale locks (`pipeline.lock`) автоматически если pid мёртв.
- [ ] A5. В `/api/methodist/mo/health` и UI «Вчера» явно показывать `partial` + `reasons` (`llm_queue_pending`).
- [ ] A6. Handoff + smoke: freshness, daily-report за вчера, одна карточка МО.

Критерий выхода A: 2 утра подряд lag<=1 или понятный алерт <15 мин после fail.

### Фаза B - Docker + вынос worker (1-2 недели)

Цель: Mac только fallback.

- [ ] B1. Добавить `Dockerfile.web` (uvicorn `rag_server:app`, non-root, health `/health/live`).
- [ ] B2. Добавить `Dockerfile.worker` (pipeline + publish deps: pandas/sqlite/ssh или лучше без ssh).
- [ ] B3. `docker-compose.yml` для локали: web + volume `medical_exams` + optional worker one-shot.
- [ ] B4. CI: build+smoke web image на PR (без PHI).
- [ ] B5. Выбрать primary worker host (решение владельца):
  - **B5a (быстрее):** маленький always-on VPS / Render Background Worker рядом с диском;
  - **B5b (правильнее к GCP):** Cloud Run Job + GCS.
- [ ] B6. Перенести расписание 06:00 Europe/Minsk на cloud scheduler (cron на VPS или Cloud Scheduler).
- [ ] B7. Publish путь: worker пишет прямо в shared storage (disk или GCS→disk), без SSH с ноутбука.
- [ ] B8. Mac launchd перевести в `fallback-only` (ручной `publish` / disaster recovery).
- [ ] B9. Документация: обновить `docs/deploy/persistent_disk.md` и короткий runbook «если worker упал».

Критерий выхода B: 5 рабочих дней подряд успешный cloud worker; Mac можно выключить на ночь без потери «Вчера».

### Фаза C - опциональный GCP (по решению владельца, 1-3 недели)

Делать **только после** стабильной фазы B.

- [ ] C1. GCP project + регион (предпочтительно EU, если нет юр. ограничений).
- [ ] C2. Artifact Registry, Cloud Build из `main` (тот же SHA-guard, что сейчас).
- [ ] C3. Cloud Run service = web; Cloud Run Job = MO daily.
- [ ] C4. GCS bucket `protocol-mo-data` (raw/reports/secure/public); IAM least privilege.
- [ ] C5. Secret Manager: `METHODIST_TOKEN`, DB password, Gemini keys.
- [ ] C6. Cloud Scheduler `0 6 * * *` timezone `Europe/Minsk`.
- [ ] C7. Cutover: DNS / домен с Render на Cloud Run (или оставить Render web, GCP только worker).
- [ ] C8. Rollback plan: 1 клик назад на Render image + последний snapshot warehouse.

Рекомендация по умолчанию: **сначала B5a или B5b worker, web оставить на Render**. Полный cutover web→GCP - отдельное решение после 2 недель зелёных метрик.

---

## 7. Решение «Docker / Render / GCP» (зафиксировать)

| Вопрос | Решение v1 |
|---|---|
| Нужен ли Docker? | Да, с фазы B; обязателен для воспроизводимости |
| Уходить с Render web сейчас? | Нет |
| Нужен ли GCP? | Опционально; обязателен только если нужен cloud worker без VPS и/или EU SLA |
| Что чинить первым? | partial LLM queue + publish token/SSH (фаза A) |
| Можно ли параллельно BI UI? | Да, но не трогать `publish_mo_to_render.py`, launchd, warehouse schema без владельца этого плана |

---

## 8. Владение файлами (anti-conflict)

Этот план владеет:

- `scripts/run_mo_daily_launchd.sh`
- `scripts/publish_mo_to_render.py`
- `clinical_knowledge/mo_publish.py` (если есть)
- новые `Dockerfile*`, `docker-compose.yml`
- docs под `docs/deploy/*` связанные с MO data plane

Не пересекать без согласования:

- `frontend/web/shared/mo-app.js` (кроме явного partial-баннера из A5);
- scorer v4 / rubric MZ планы;
- GitHub release workflow (уже completed).

Ветка задачи (пример):

```bash
scripts/ops/git_task_start.sh mo-runtime-stabilization --pc=pc1 \
  --branch=codex/mo-runtime-stabilization-agent1-pc1
```

---

## 9. Риски

| Риск | Митигация |
|---|---|
| Worker в облаке не достучится до MariaDB (VPN/IP) | Allowlist IP / SSH tunnel / ETL на машине с VPN, артефакты в GCS |
| Publish затрёт CRM в проде | Сохранить текущий merge-скрипт без CRM overwrite |
| PHI в Docker/CI логах | Запрет печати clinical text; secrets scanning |
| Двойной primary (Mac + cloud) пишут одновременно | Один leader lock в state; Mac только fallback |
| GCP стоимость | Считать до C1; web можно оставить на Render |
| Долгий LLM drain оставляет eternal `partial` | Политика A1: coverage gate vs LLM advisory |

---

## 10. Оценка трудозатрат

| Фаза | Оценка | Блокер |
|---|---|---|
| A | 3-5 рабочих дней | токен launchd, политика LLM queue |
| B | 5-10 рабочих дней | выбор VPS vs Cloud Run Job, доступ к MIS |
| C | 5-15 рабочих дней | решение владельца + GCP billing |

---

## 11. Первая безопасная команда

```bash
git status --short --branch
git fetch --prune origin
git rev-list --left-right --count origin/main...HEAD
# затем только фаза A в отдельном worktree:
scripts/ops/git_task_start.sh mo-runtime-stabilization --pc=pc1 \
  --branch=codex/mo-runtime-stabilization-agent1-pc1
```

Не начинать с создания GCP-проекта и не останавливать Render, пока не закрыта фаза A.

---

## 12. Definition of Done

План считается выполненным (без обязательного GCP), когда:

1. фазы A и B закрыты по чеклистам;
2. 5 дней подряд «Вчера» доступно в проде к 07:00 Минск;
3. Mac не является primary;
4. есть Docker web+worker и короткий disaster-recovery runbook;
5. handoff записан в `docs/reports/YYYY-MM-DD-handoff-mo-runtime-stabilization.md`.

Фаза C - отдельный go/no-go владельца после DoD фаз A+B.
