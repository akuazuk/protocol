# Handoff: accounts admin on methodist МО Аналитика tab

Дата: 2026-08-09

## Repo / branch

- branch: `cursor/auth-accounts-admin-tab-pc1`
- worktree: `/private/tmp/protocol-task-auth-accounts-admin-tab-pc1`
- PR: https://github.com/akuazuk/protocol/pull/89
- plan: `docs/plans/2026-08-09-auth-accounts-unify-v1.md`
- BUILD_VERSION: `2026-08-09-065903Z-auth-accounts-admin`

## Сделано

- Вкладка «МО Аналитика» в режиме методиста: CRUD учёток вместо сводки-заглушки.
- `crm_app_user` / session: роль, `mo_access` (reports|full), `reports_min_date`.
- Вход в `/methodist/mo` логин/пароль + токен методиста.
- Тесты `tests/test_mo_app_accounts.py`.

## Не сделано

- Deploy (ждать merge + Render/GCE).
- Аудит изменений учёток.
- Bootstrap admin из env.

## Следующая команда

После зелёного CI: merge #89 → production release → smoke создания учётки.
