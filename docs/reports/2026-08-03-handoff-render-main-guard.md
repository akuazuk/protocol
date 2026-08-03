# Handoff: Render main release guard

Дата: 2026-08-03

## Git

- Repo: `akuazuk/protocol`
- Base: `origin/main` at `be172d73e7f57032d6a35539c8389560812b3307`
- Branch: `cursor/render-main-guard-agent1-pc1`
- Implementation commit: `30316a87`
- PR: https://github.com/akuazuk/protocol/pull/7
- Build: `2026-08-03-r21-render-main-guard`

## Сделано

- Включена GitHub branch protection для `main`: обязателен PR, защита действует для
  администратора, запрещены force-push и удаление ветки.
- Старые task HEAD promote scripts переведены в fail-closed режим.
- Единственный release wrapper принимает только точный текущий SHA `origin/main`.
- Низкоуровневые Render wrappers больше не используют локальный HEAD по умолчанию.
- Добавлены общие правила в `AGENTS.md` и `.cursor/rules/production-release.mdc`.
- `/api/version` дополнен полем `git_commit`.

## Проверки

- `tests/test_release_guards.py` и `tests/test_mo_frontend_structure.py`: 14 passed.
- `scripts/ops/smoke_repo_layout.sh`: `SMOKE_OK`.
- `bash -n` изменённых scripts: успешно.
- Dry-run точного `origin/main`: успешно.
- Dry-run unmerged `7988ae78`: корректно заблокирован.
- Branch protection API: PR required, admins enforced, linear history, no force-push/delete.

## Production

На момент handoff production остаётся на `be172d73` /
`2026-08-03-r19-rubric-handoff-workflow`, что соответствует текущему `origin/main`.
Deploy `r21` разрешён только после merge PR #7.

## Следующая безопасная команда

```bash
git fetch origin
scripts/ops/render_release_main.sh --commit="$(git rev-parse origin/main)"
```

Не запускать команду до merge PR #7 и фиксации его merge SHA.
