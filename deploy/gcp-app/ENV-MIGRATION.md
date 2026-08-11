# Перенос env: Render → GCE staging

Секреты **никогда** не коммитить. Источник истины на GCP - **Secret Manager**.

| Файл / секрет | Содержимое |
|--|--|
| SM `kravira-db-password` | пароль Marina |
| SM `google-api-key`, `google-api-key-2`, `gemini-api-key*`, `generative-language-api-key` | LLM keys |
| SM `methodist-token` | `METHODIST_TOKEN` |
| `/opt/protocol/.env.gcp-public` | non-secret web (RAG_*, PORT, ICD flags, model names) |
| `/opt/protocol/.env.gcp-staging` | **assembled** runtime env для `docker --env-file` (public + SM) |
| `/opt/protocol/.env.mis` | non-secret MIS DSN (host/port/user), owner=`pavel` |

## Команды

```bash
# MIS password + DSN
bash deploy/gcp-app/push_mis_env.sh
bash deploy/gcp-app/mis_sql_smoke_on_gce.sh

# Gemini / methodist → SM + assemble web env (+ optional container restart)
bash deploy/gcp-app/push_web_secrets.sh --restart-web

# Full app deploy (uploads SM + public, builds image, assembles env, runs container)
SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh
```

Night extract (только GCE):

```bash
bash deploy/gcp-app/install_night_cron.sh --remote
# cron UTC: 02:00 main, 03:00 retry; crontab user = pavel
```

Loader MIS: `deploy/gcp-app/load_mis_env.sh`
Assembler web: `deploy/gcp-app/assemble_web_env_from_sm.sh`

VM SA `…-compute@developer.gserviceaccount.com` → `roles/secretmanager.secretAccessor`
на каждый секрет.

## Важно

- Cron user = `pavel`. Env-файлы: `chown pavel` (`GCE_OPS_USER`), иначе `Permission denied`.
- В `.env.mis` и `.env.gcp-public` **не** должно быть паролей/API keys.
- Runtime `.env.gcp-staging` собирается на VM из SM перед стартом контейнера (нужен docker env-file).

## Не переносить слепо

- Dump Render env целиком.
- Секреты в git / Docker image COPY.
- PHI paths как git-файлы.
