# Handoff: Aug10 night fail + MIS password → Secret Manager

Дата: 2026-08-11
Branch: `cursor/night-env-perm-cron-user-pc1`
Primary: `https://protocol.kravira.by`

## Что случилось ночью (10.08 → cron 11.08 02:00 UTC)

**Не** «Google не подключился к Marina». SQL с GCE работает.

Причина: cron user = `pavel`, файл `/opt/protocol/.env.mis` был `600`
owner `pavelkuzauka` (после deploy/push под другим SSH-логином) →
`Permission denied` на `source`. Retry и check упали так же.
State `gce_night_2026-08-10.json` не появился → в UI нет выгрузки.

Ручной re-run после `chown pavel:pavel`: **success**, 569 rows / 449 cases.

## Что сделано сейчас

1. Пароль MIS в **Secret Manager** `kravira-db-password` (project `protocol-home-e1`).
2. VM SA `…-compute@developer.gserviceaccount.com` → `secretAccessor`.
3. `.env.mis` на диске - **только** host/port/user/timeouts, owner `pavel`, без пароля.
4. Loader `deploy/gcp-app/load_mis_env.sh`; night/smoke/push/deploy обновлены.
5. Mac `com.kravira.mis-*` launchd **выгружен и `.disabled-…`**.

## Smoke

- `mis_sql_smoke_on_gce.sh` → `SQL_OK`, `mis_protocol_max_date=2026-08-10`
- Cron-like PATH + SM access OK (`/usr/bin/gcloud`)

## Не сделано (следующий шаг)

- Gemini / `METHODIST_TOKEN` ещё в `.env.gcp-staging` → тоже в Secret Manager.
- Полный `deploy_to_gce.sh` app image не обязателен для ночи (скрипты уже на VM).

## Next

```bash
# после merge:
# (опционально) SYNC_PROTOCOL_CORPUS=0 bash deploy/gcp-app/deploy_to_gce.sh
curl -fsS https://protocol.kravira.by/api/version
```
