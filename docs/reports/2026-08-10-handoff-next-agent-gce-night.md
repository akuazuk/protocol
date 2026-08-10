# Handoff: GCE night MIS merged + deployed (next agent)

Date: 2026-08-10  
Audience: другой агент / другой компьютер  
Primary: `https://protocol.kravira.by`

---

## 1. Что уже сделано (не повторять)

| Item | State |
|--|--|
| PR **#124** | **MERGED** → `origin/main` `045a8c0` |
| PR #121 / #123 | closed (superseded by #124) |
| PR **#122** (parquet plan docs) | open or merging; docs-only, не блокер ночи |
| GCE deploy | done: `/health/live` ok, `/api/version` = `2026-08-10-101544Z-night-speed-impl` |
| Night cron on VM | **02:00** main, **03:00** retry, **03:15** check (UTC) |
| MIS secrets | `/opt/protocol/.env.mis` present |
| Mac Protocol launchd | **uninstalled** (`by.protocol.mo-daily*`) |
| sql_epam Mac jobs | **untouched** (`com.kravira.mis-*` still loaded) |

Data on GCE: warehouse **2026-01-02…2026-08-09**; night smoke day **2026-08-09** success.

---

## 2. Канон (обязательно читать)

1. `AGENTS.md` - Git/PR; primary deploy = GCE, не Render.  
2. `docs/plans/2026-08-07-by-home-gcp-llm-split-v1.md` - E2 MIS с GCP.  
3. `docs/plans/2026-08-10-mo-night-speed-skip-alerts-v1.md` - workers/skip/alerts (**реализовано**).  
4. `docs/plans/2026-08-10-mo-storage-parquet-dual-write-v1.md` - parquet dual-write (**только план**).  
5. `.cursor/rules/mis-mariadb.mdc` - extract только с GCE; Mac SQL = ad-hoc.  
6. Gemini/LLM night: `deploy/gcp-llm/run_on_gce.sh` (не Mac).

---

## 3. Preflight перед любой работой

```bash
git status --short --branch
git fetch --prune origin
git rev-list --left-right --count origin/main...HEAD
gh pr list --repo akuazuk/protocol --state open
curl -fsS https://protocol.kravira.by/api/version
curl -fsS https://protocol.kravira.by/health/live
```

Грязный/отстающий checkout **не чинить** pull/rebase - новый worktree от `origin/main`:

```bash
scripts/ops/git_task_start.sh <slug> --pc=<pc-id> \
  --branch=cursor/<slug>-<agent>-<pc>
```

---

## 4. Что делать дальше (приоритет)

### P0 - утро после первой ночи (завтра)

Проверить, что cron вытащил **2026-08-10** (вчера по Минску):

```bash
gcloud compute ssh protocol-app --zone=europe-central2-a --command='
crontab -l
cat /var/data/medical_exams/state/gce_night_$(python3 -c "from datetime import datetime,timedelta; from zoneinfo import ZoneInfo; print((datetime.now(ZoneInfo(\"Europe/Minsk\")).date()-timedelta(days=1)).isoformat())").json
tail -40 /var/data/medical_exams/logs/gce-night-main.log
tail -20 /var/data/medical_exams/logs/gce-night-check.log
'
curl -fsS https://protocol.kravira.by/api/version
```

Ожидание: `status=success`, `workers=2`, при повторе - `unchanged_skip_score` ок.  
Если fail → читать `ALERT_NEEDED` / Telegram; чинить extract, не включать Mac launchd.

### P1 - опционально добить docs PR #122

Если ещё OPEN: дождаться зелёного CI, squash-merge. Не блокирует прод.

### P2 - не начинать без явного запроса владельца

- Фаза 1 parquet dual-write (`2026-08-10-mo-storage-parquet-dual-write-v1.md`)  
- Secret Manager для `KRAVIRA_DB_PASSWORD`  
- Сдвиг cron на 02:00 **Минск** (сейчас 02:00 **UTC** = 05:00 Минск)  
- Postgres вместо sqlite  

### P3 - продукт (другие планы)

Scorer / zones / navigator / auth - отдельные PR; **не** трогать параллельно `deploy/gcp-app/night_mis_pipeline.sh` без очереди.

---

## 5. Операционные команды

```bash
# SQL smoke Marina from GCE
bash deploy/gcp-app/mis_sql_smoke_on_gce.sh

# Push MIS password again (from Mac sql_epam/.env)
bash deploy/gcp-app/push_mis_env.sh

# Manual night for one day
gcloud compute ssh protocol-app --zone=europe-central2-a --command='
export MO_NIGHT_DAY=YYYY-MM-DD MO_NIGHT_WITH_LLM=0 MO_DAILY_WORKERS=2
bash /opt/protocol/deploy/gcp-app/night_mis_pipeline.sh main
'

# Force full re-score
# MO_NIGHT_FORCE=1 bash .../night_mis_pipeline.sh main

# App redeploy after merge (coordinator only)
git fetch origin && git rev-parse origin/main
SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh
curl -fsS https://protocol.kravira.by/api/version   # must match BUILD_VERSION
```

---

## 6. Нельзя

- Запускать Gemini/`grade_kz_llm` с Mac.  
- Снова ставить Protocol Mac launchd для SQL extract.  
- Push в `main` напрямую; promote task-HEAD.  
- Считать Render primary (`protocol-bimy` = backup).  
- Печатать `KRAVIRA_DB_PASSWORD` / токены / PHI.  
- Ломать cron entrypoint `main|retry` без плана.

---

## 7. Файлы «горячие» (не параллелить)

- `deploy/gcp-app/night_mis_pipeline.sh`  
- `deploy/gcp-app/install_night_cron.sh`  
- `deploy/gcp-app/score_inbound_day.sh`  
- `deploy/gcp-app/deploy_to_gce.sh`  
- `/opt/protocol/.env.mis` на VM  
- `scripts/export_mis_protocol_month.py`  

---

## 8. Одна безопасная следующая команда

```bash
# Завтра утром (или сейчас для dry-check вчерашнего статуса):
gcloud compute ssh protocol-app --zone=europe-central2-a --quiet --command='crontab -l; ls -la /var/data/medical_exams/state/gce_night_*.json | tail -5; tail -30 /var/data/medical_exams/logs/gce-night-main.log'
```

Если ночь ещё не прошла - ждать 02:00 UTC; не гонять Mac extract.
