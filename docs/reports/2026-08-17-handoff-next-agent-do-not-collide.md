# Handoff для другого агента / компьютера (не перепутать работу)

Дата: 2026-08-17
Владелец этой волны: Cursor pc1
Репозиторий: `akuazuk/protocol`
Канон: `origin/main` `2e27373` (#156)
**Не работать** в checkout `/Users/pavel/CURSOR/Protocol/protocol` на `main`: он грязный и отстаёт на 14 коммитов от `origin/main` (последний pull там `#140` `c263057`, 2026-08-11). Не делать pull/rebase/reset в том каталоге. Новый clean worktree от свежего `origin/main`.

Primary: `https://protocol.kravira.by`
Прод сейчас: `BUILD_VERSION` `2026-08-14-123546Z-kp-golden-40` (merge `#155` `b6c7381`, deploy 2026-08-17). `#156` только docs, в runtime не выкладывался.

---

## Жёсткий запрет прямо сейчас

На GCE идёт **полный parse инструкций ЛС**. Рестарт `protocol-web` или `deploy_to_gce.sh` **убьёт** job.

| | |
|--|--|
| VM | `protocol-app` / `europe-central2-a` |
| Job | `bash /opt/protocol/deploy/gcp-app/rceth_sync_job.sh` (host pid около 403571) |
| В контейнере | `python /app/scripts/rceth_sync_run.py --data-root /var/data/rceth parse` |
| Режим | `RCETH_MODE=full` `RCETH_LIMIT=0` `RCETH_PARSE=1` |
| Данные | `/var/data/rceth/` (не в git) |
| На 2026-08-17 ~11:52 UTC | parse **510 / 3017**, PDF уже **3017**, labels **510**, download fail 0 |
| Watchdog | cron `*/10` уже включён; weekly cron **не** включать |

**Не делать, пока `status.json` не `done`:**

- `deploy_to_gce.sh`, restart/stop `protocol-web`, `docker rm`, reboot VM;
- второй `run_rceth_full_on_gce.sh` / `rceth_sync_job.sh` (ложный `pgrep` в launcher ловит сам SSH; не считать это «уже запущено» и не убивать живой parse);
- менять `/var/data/rceth/_sync/last_job.env` на `LIMIT=50` или `SKIP_DOWNLOAD=1`;
- `RCETH_PARSE_FORCE=1` (перетрёт уже написанные labels);
- включать `--enable-weekly` до успешного full;
- трогать night MIS cron / `/opt/protocol/.env.mis` / Secret Manager.

Проверка без вреда:

```bash
gcloud compute ssh protocol-app --zone=europe-central2-a --command='
python3 -c "import json; d=json.load(open(\"/var/data/rceth/_sync/status.json\")); print(d.get(\"status\"), d.get(\"phase\"), d.get(\"progress\"), d.get(\"updated_at\"))"
pgrep -af "bash /opt/protocol/deploy/gcp-app/rceth_sync_job.sh" || true
'
```

Parse - это не оценка врача. По каждому `_s.pdf`: текст (pypdf / PyMuPDF / OCR) → секции 4.1/4.2/4.3 (+ 4.4/4.5) → `/var/data/rceth/labels/{reg_id}.json`. `parse_ok` = нашлись 4.1+4.2+4.3. Остаток parse ещё много часов.

---

## Что в `origin/main` после последнего pull на этом Mac (`#140`)

14 squash-merge, от старых к новым:

| SHA | PR | Суть |
|--|--|--|
| `239ec02` | #141 | Night self-heal owner `.env` на GCE, extract не встаёт |
| `d34fee0` | #142 | Daily сверка КП МЗ, changed-only, recency |
| `b17df10` | #143 | Night KP sync через `protocol-web` |
| `afe0fdf` | #144 | Вкладка КП: даты, итоги периода, графики |
| `e24ee66` | #145 | History continuity + отбор poor на deep run |
| `2d2243e` | #147 | Bulk `patient_key` backfill без pydantic на host |
| `8d69404` | #149 | КП: диагноз→МКБ→жалобы, content index, I84 не аневризма, нет specialty-filler |
| `cf71037` | #146 | Rceth: crawl/download/parse, вкладка «Инструкции ЛС», пилот |
| `6016231` | #151 | Rceth UI: не вечный running после смерти job |
| `0a15236` | #152 | КП: возраст на дату визита, in-force, омнибус-демоут |
| `6aab2ba` | #153 | Rceth full launcher + watchdog |
| `4f0d873` | #154 | КП prefilter + ICD из содержания (K21/K30/E03) |
| `b6c7381` | #155 | Golden 40 КП (возраст + дата визита). **Это в проде.** |
| `2e27373` | #156 | Docs: метрики CSV-прогона КП. **Не деплоить отдельно.** |

Планы:

- КП точность: `docs/plans/2026-08-14-mo-kp-suggest-accuracy-v2.md` (шаги 1-9 сделаны).
- ЛС: `docs/plans/2026-08-14-rceth-drug-labels-mo-v1.md` (сейчас шаг C/G: full parse).
- КП daily: `docs/plans/2026-08-13-minzdrav-kp-daily-sync-v2.md`.
- History deep: `docs/plans/2026-08-14-mo-history-continuity-deep-run-v1.md`.

Уже есть узкий handoff КП: `docs/reports/2026-08-17-handoff-kp-suggest-accuracy.md`.

---

## Что сделано в этой сессии (pc1), чего нет в git как runtime

- CSV-прогон КП 26.07-13.08 на GCE: 7605 clinical, hit 71.2%, возраст 98.8%, омнибус top-1 568 (ЛОР 2017 ×426). Отчёт без PHI: `/var/data/medical_exams/reports/kp_suggest_eval_2026-07-26_08-13.json`.
- Deploy `#155` на GCE 2026-08-17, smoke ok.
- Полный Rceth: crawl+download закончены (7352 в манифесте, 3017 `_s`, 2967 новых + 50 skip пилота, fail 0). Parse идёт.
- `run_rceth_full_on_gce.sh` с Mac дал ложный `ERROR: rceth_sync_job.sh already on host` (pgrep ловит argv SSH). Job стартовали напрямую `nohup env RCETH_LIMIT=0 ... bash /opt/protocol/deploy/gcp-app/rceth_sync_job.sh`. Не запускать второй.

50 инструкций раньше - не сбой, пилот `RCETH_LIMIT=50`.

---

## Открытые чужие PR - не трогать и не rebase

| PR | Тема |
|--|--|
| **#148** | billed Gemini key, history deep layer C |
| #120 #113 #97 #77 | старые docs/calibration, draft |

Не пересекаться с `#148` по Gemini/history-deep.

---

## Каталоги: кто чем владеет

| Можно брать (после parse `done`, отдельная ветка) | Не трогать, пока Rceth parse жив / без координации |
|--|--|
| КП омнибус: `protocol_match.py`, `kp_validity.py`, `case_protocol_suggest.py`, golden fixture | `clinical_knowledge/rceth_sync/**` |
| Новый план/метрики КП | `deploy/gcp-app/rceth_sync*.sh`, `run_rceth_full_on_gce.sh` |
| | `scripts/rceth_sync_run.py` |
| | `/var/data/rceth/**` на GCE |
| | `frontend/web/shared/mo-app.js` rceth-блоки и `mis-kz-quality.html` якоря `rceth-sync-*` |
| | `rag_server.py` BUILD_VERSION, если собираешься деплоить |
| | Rceth / history-deep / Rceth cron |

`rag_server.py` и `mo-app.js` уже делили несколько PR. Второй агент не правит их в одном PR с Rceth и не деплоит «заодно».

Night MIS, Gemini с Mac, прямой push в `main` - по `AGENTS.md`. Render не primary.

---

## Что делать дальше (по приоритету)

1. **Ничего не деплоить.** Дождаться parse `done` (ещё много часов; watchdog сам resume).
2. Когда `done`: снять агрегаты без PHI (`parse_ok`, `needs_human`, `empty_text`, `failed`) в `docs/reports/` и отметить шаг C в плане Rceth. **Не** печатать тексты инструкций / INN-списки в git.
3. Weekly cron (`install_rceth_cron.sh --remote --enable-weekly`) - только после успешного full.
4. Потом по плану Rceth: D (бренд→МНН), F (shadow findings). Primary findings - решение владельца.
5. Параллельно, **без деплоя и без Rceth-файлов**, можно КП: ужесточить омнибус ЛОР 2017 / урология 2011 (цель ≤5% omnibus top-1). План v2, шаг «следующая команда».
6. Мелкий техдолг (отдельный PR, merge не деплоить до конца parse): починить pgrep в `run_rceth_full_on_gce.sh`.

Безопасный старт своей задачи:

```bash
git fetch --prune origin
scripts/ops/git_task_start.sh <slug> --pc=<pc-id> \
  --branch=cursor/<slug>-<agent>-<pc>
# base только origin/main 2e27373 или новее после своего fetch
```

Одна безопасная команда прямо сейчас (только чтение):

```bash
gcloud compute ssh protocol-app --zone=europe-central2-a --command='python3 -c "import json; print(json.load(open(\"/var/data/rceth/_sync/status.json\")).get(\"progress\"))"'
```
