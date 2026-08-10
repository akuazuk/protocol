# Перенос env: Render → GCE staging

Секреты **никогда** не коммитить. На staging:

- `/opt/protocol/.env.gcp-staging` - web / Gemini / methodist (создаёт `deploy_to_gce.sh`)
- `/opt/protocol/.env.mis` - Marina MariaDB (E2; `deploy_to_gce.sh` + `push_mis_env.sh`)

## Что уже работает

`deploy/gcp-app/deploy_to_gce.sh` копирует с Mac:

Web (из локального `.env`):

- `GOOGLE_API_KEY` / `GEMINI_API_KEY` (+ `_2`, `GENERATIVE_LANGUAGE_API_KEY`)
- `METHODIST_TOKEN`, модели Gemini (если заданы)
- staging defaults: `MO_DATA_ROOT`, `RAG_STARTUP_MODE=manifest`, …
- ICD (фаза 3/4): `MO_ICD_*` defaults

MIS (из `.env` или `~/CURSOR/sql_epam/.env` → remote `.env.mis`):

- `KRAVIRA_DB_PASSWORD` (обязателен)
- `KRAVIRA_DB_HOST` / `PORT` / `USER` / `NAME` (defaults Kravira)
- `MIS_DB_CONNECT_TIMEOUT`, `MIS_DB_READ_TIMEOUT`
- `RUN_HOST=gcp`

MIS **не** кладётся в `protocol-web` по умолчанию (граница с llm/web).
Только extract/smoke читают `.env.mis`. Для отладки:
`INCLUDE_MIS_IN_WEB_ENV=1 bash deploy/gcp-app/deploy_to_gce.sh`.

Быстрый push секрета без полного деплоя:

```bash
bash deploy/gcp-app/push_mis_env.sh
bash deploy/gcp-app/mis_sql_smoke_on_gce.sh
```

Ночной extract (только GCE, Mac launchd off):

```bash
bash deploy/gcp-app/install_night_cron.sh --remote
# cron UTC: 02:00 main, 03:00 retry
```

## Канон дальше (ближе к прод-cutover)

### 1. Снять список с Render (имена, без печати секретов в чат)

```bash
scripts/ops/render_env.sh list          # имена
# при необходимости локально:
scripts/ops/render_env.sh list --show-values > /tmp/render-env.local
chmod 600 /tmp/render-env.local
```

### 2. Разделить ключи

| Класс | Примеры | Куда на GCP |
|--|--|--|
| LLM | `GOOGLE_API_KEY`, `GEMINI_*` | Secret Manager → mount/env в container |
| Auth UI | `METHODIST_TOKEN`, expert tokens | Secret Manager |
| Paths | `MO_DATA_ROOT`, `RAG_*` | non-secret env / compose |
| Render-only | `RENDER_API_KEY`, `RENDER_SERVICE_ID` | **не** на GCE |
| MIS | `KRAVIRA_DB_PASSWORD` (+ host/port/user) | **`.env.mis` на GCE (E2)**; дальше Secret Manager |

### 3. Положить в Secret Manager (project `protocol-home-e1`)

```bash
gcloud config set project protocol-home-e1
# пример одного секрета (значение из stdin, не из argv):
printf '%s' "$GOOGLE_API_KEY" | gcloud secrets create google-api-key --data-file=-
# обновление:
printf '%s' "$GOOGLE_API_KEY" | gcloud secrets versions add google-api-key --data-file=-
```

Выдать VM SA доступ `roles/secretmanager.secretAccessor`, в run/container -
подтягивать через startup или `gcloud secrets versions access`.

### 4. Пока без SM (staging)

Повторять `bash deploy/gcp-app/deploy_to_gce.sh` с актуального Mac `.env`
(+ `sql_epam/.env` для MIS). Файлы на VM: `chmod 600`, не в git, не в Docker image.

### 5. После смены ключей

```bash
bash deploy/gcp-app/deploy_to_gce.sh
# или только MIS:
bash deploy/gcp-app/push_mis_env.sh && bash deploy/gcp-app/mis_sql_smoke_on_gce.sh
curl -fsS https://protocol.kravira.by/api/version
```

## Не переносить слепо

- Весь dump Render env: там есть Render-специфичное и лишнее.
- MIS password в образ llm / в git.
- PHI paths как git-файлы.
