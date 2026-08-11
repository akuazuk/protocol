# Handoff: web secrets to Secret Manager

Дата: 2026-08-11
Branch: `cursor/web-secrets-sm-pc1`
Merged earlier: #135 (MIS password SM)

## Done

- SM secrets: `google-api-key`, `gemini-api-key`, `methodist-token`,
  `telegram-bot-token`, `telegram-chat-id`, `render-api-key` (+ MIS already)
- `/opt/protocol/.env.gcp-public` = allowlist only (no API keys)
- `assemble_web_env_from_sm.sh` builds `.env.gcp-staging` for docker
- `push_web_secrets.sh` / `deploy_to_gce.sh` wired
- Deployed: `/api/version` = `2026-08-11-062430Z-web-secrets-sm`

## Ops note

Первый push ошибочно клал весь `.env` в public (Telegram/Render).
Исправлено allowlist-ом; рекомендуется ротация этих двух ключей.

## Next

Merge PR. Rotate Telegram/Render when convenient.
