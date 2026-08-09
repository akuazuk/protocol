# Handoff: auth accounts P0 (SSO + expert redirect)

Дата: 2026-08-09

## Repo / branch

- repo: `akuazuk/protocol`
- branch: `cursor/auth-accounts-p0-pc1`
- worktree: `/private/tmp/protocol-task-auth-accounts-p0-pc1`
- base: `origin/main`
- plan: `docs/plans/2026-08-09-auth-accounts-unify-v1.md`

## Сделано (P0)

- Кабинет методиста пишет/читает токен и reviewer в `localStorage` + `sessionStorage`.
- `mo-api.js`: `setToken`/`clearToken`, sync storage, на МО не предпочитает leftover expert-session.
- `mo-app.js`: 403 «нет прав» не выкидывает на login; auth-failure - да.
- `/methodist/expert*` → 302 `/methodist/mo`.
- Тесты `tests/test_auth_accounts_p0.py`.

## Не сделано

- P1 login/password учётки методиста.
- P2 admin CRUD.
- P3 полное удаление expert API/схемы.

## Тесты

```bash
.venv/bin/pytest tests/test_auth_accounts_p0.py tests/test_workspace_routes.py tests/test_mo_frontend_structure.py -q
```

## Deploy

- Не merged. Production не трогать до PR merge + Action.
- `BUILD_VERSION`: смотреть `rag_server.py` на ветке.

## Следующая команда

После merge/smoke - task-ветка P1 от свежего `origin/main`.

## Не трогать параллельно

- `frontend/web/shared/mo-api.js`, `mo-app.js`
- methodist login block в `frontend/web/doctor/index.html`
- expert routes в `rag_server.py`
