# Render main-branch deploy rule

Дата: 2026-07-29  
Ветка: `codex/main-sync`

## Проблема

Render деплоит из `main`, но рабочие коммиты могли оставаться только в `codex/main-sync`.
В этом случае прод оставался на старом SHA.

## Что добавлено

- Новый safe-script: `scripts/ops/render_promote_main.sh`
  - проверяет clean tree;
  - проверяет fast-forward возможность (`origin/main` является предком текущего `HEAD`);
  - делает `git push origin HEAD:main`;
  - подтверждает, что `origin/main` получил текущий SHA;
  - по умолчанию ждёт совпадения `/api/version` с локальным `BUILD_VERSION`.
- Обновлены:
  - `docs/deploy/multi-machine-git-deploy-runbook.md`
  - `README.md`
  - `.cursor/rules/git-push.mdc`

## Стандартный сценарий

```bash
scripts/ops/render_promote_main.sh --prod-url=https://protocol-bimy.onrender.com
```
