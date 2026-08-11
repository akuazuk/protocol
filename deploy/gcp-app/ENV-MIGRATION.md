# Перенос env: Render → GCE staging

Секреты **никогда** не коммитить. На staging:

- `/opt/protocol/.env.gcp-staging` - web / Gemini / methodist (создаёт `deploy_to_gce.sh`; SM - следующий шаг)
- `/opt/protocol/.env.mis` - **только non-secret** DSN (host/port/user/timeouts), owner=`GCE_OPS_USER=pavel`
- **Secret Manager** `kravira-db-password` (project `protocol-home-e1`) - пароль Marina

## Что уже работает

`deploy/gcp-app/deploy_to_gce.sh` / `push_mis_env.sh`:

Web (из локального `.env` → remote `.env.gcp-staging`):

- `GOOGLE_API_KEY` / `GEMINI_API_KEY` (+ `_2`, `GENERATIVE_LANGUAGE_API_KEY`)
- `METHODIST_TOKEN`, модели Gemini (если заданы)
- staging defaults: `MO_DATA_ROOT`, `RAG_STARTUP_MODE=manifest`, …
- ICD (фаза 3/4): `MO_ICD_*` defaults

MIS:

- Пароль → `gcloud secrets … kravira-db-password` (VM SA = secretAccessor)
- Host/port/user/name/timeouts → `.env.mis` **без** `KRAVIRA_DB_PASSWORD`
- Night/smoke: `source deploy/gcp-app/load_mis_env.sh` (SM → env)

MIS **не** кладётся в `protocol-web` по умолчанию.
Для отладки: `INCLUDE_MIS_IN_WEB_ENV=1` (не рекомендуется).

Быстрый push без полного деплоя:

```bash
bash deploy/gcp-app/push_mis_env.sh
bash deploy/gcp-app/mis_sql_smoke_on_gce.sh
```

Ночной extract (только GCE, Mac launchd off):

```bash
bash deploy/gcp-app/install_night_cron.sh --remote
# cron UTC: 02:00 main, 03:00 retry; user crontab = pavel
```

Важно: cron user = `pavel`. Файлы env всегда `chown pavel:pavel` (`GCE_OPS_USER`), иначе
`Permission denied` на `source` (как 2026-08-11 ночь за 10.08).

## Канон дальше

### 1. Разделить ключи

| Класс | Примеры | Куда на GCP |
|--|--|--|
| MIS password | `KRAVIRA_DB_PASSWORD` | **Secret Manager `kravira-db-password`** |
| MIS DSN public | host/port/user | `.env.mis` (600, owner pavel) |
| LLM | `GOOGLE_API_KEY`, `GEMINI_*` | Secret Manager → mount/env в container (TODO) |
| Auth UI | `METHODIST_TOKEN` | Secret Manager (TODO) |
| Paths | `MO_DATA_ROOT`, `RAG_*` | non-secret env |
| Render-only | `RENDER_API_KEY` | **не** на GCE |

### 2. Secret Manager (project `protocol-home-e1`)

```bash
gcloud config set project protocol-home-e1
# пароль MIS (stdin, не argv):
printf '%s' "$KRAVIRA_DB_PASSWORD" | gcloud secrets versions add kravira-db-password --data-file=-
# или:
bash deploy/gcp-app/push_mis_env.sh
```

VM SA `…-compute@developer.gserviceaccount.com` → `roles/secretmanager.secretAccessor`.

### 3. После смены ключей

```bash
bash deploy/gcp-app/push_mis_env.sh && bash deploy/gcp-app/mis_sql_smoke_on_gce.sh
# полный app deploy:
SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh
curl -fsS https://protocol.kravira.by/api/version
```

## Не переносить слепо

- Весь dump Render env.
- MIS password в образ / git / `.env.mis`.
- PHI paths как git-файлы.
