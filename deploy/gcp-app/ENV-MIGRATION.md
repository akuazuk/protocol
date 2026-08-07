# Перенос env: Render → GCE staging

Секреты **никогда** не коммитить. На staging сейчас: файл
`/opt/protocol/.env.gcp-staging` на VM (создаёт `deploy_to_gce.sh` из локального `.env`).

## Что уже работает

`deploy/gcp-app/deploy_to_gce.sh` копирует с Mac `.env` только ключи:

- `GOOGLE_API_KEY` / `GEMINI_API_KEY` (+ `_2`, `GENERATIVE_LANGUAGE_API_KEY`)
- `METHODIST_TOKEN`, модели Gemini (если заданы)
- staging defaults: `MO_DATA_ROOT`, `RAG_STARTUP_MODE=manifest`, …

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
| MIS | `KRAVIRA_DB_PASSWORD` | **не** на GCE в E1 (остаётся Mac bridge) |

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

Повторять `bash deploy/gcp-app/deploy_to_gce.sh` с актуального Mac `.env`.
Файл на VM: `chmod 600`, не в git, не в Docker image layers как COPY.

### 5. После смены ключей

```bash
bash deploy/gcp-app/deploy_to_gce.sh
curl -fsS http://34.118.21.47:8000/api/version
```

## Не переносить слепо

- Весь dump Render env: там есть Render-специфичное и лишнее.
- Пароль МИС на VM в эпохе E1.
- PHI paths как git-файлы.
