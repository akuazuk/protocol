# Handoff: упаковка лабораторных справочников в GCE image

Дата: 2026-09-06.
Repo: `akuazuk/protocol`.
Branch: `cursor/mo-lab-assets-packaging-agent1-pc1`.
Worktree: `/private/tmp/protocol-task-mo-lab-assets-packaging-pc1`.
Base: `8270b8746e9f3b2e48f9d91feedb87cd6bb8113c`.

## Причина

После merge #206 production image не содержал:

- `/app/data/lab_canons/lab_reference_ranges.json`;
- `/app/data/lab_canons/lab_test_canons.json`.

Каталог отсутствовал одновременно в allowlist `git archive` и в Dockerfile.
Поэтому локальные тесты лабораторного evaluator не доказывали его
работоспособность в release artifact.

## Реализация

- `data/lab_canons` добавлен в точный archive релизного SHA.
- Оба JSON копируются в GCE image.
- Docker build запускает `verify_lab_assets.py`: проверяет непустые ranges/panels
  и synthetic SQLite -> `evaluate_lab_for_case`.
- После сборки, до остановки работающего контейнера, deploy повторяет verifier
  отдельным `docker run` точного SHA image.
- Primary flags, справочники, clinical weights, данные и production env не менялись.

## Проверки

- verifier локально: `ranges=8`, `panels=17`, один shadow finding;
- 35 тематических lab tests passed;
- `ruff`, `bash -n`, `git diff --check` passed;
- локальный Docker отсутствует, поэтому точный image build проверяет CI
  `docker-images` на опубликованном HEAD;
- release acceptance требует повторного verifier внутри SHA image и проверки
  обоих JSON в работающем контейнере.

## Состояние release

На начало задачи `origin/main`:
`8270b8746e9f3b2e48f9d91feedb87cd6bb8113c`.

Production:
`a592d588fdd7eb428161024ad13e4e3948bb3754`,
version `2026-09-06-073651Z-deploy-lock-allowlist`.
Оба JSON в работающем container отсутствуют. Deploy этой ветки не выполнялся.

PR #195 имеет hard overlap по Dockerfile и красный CI. Последовательность
зафиксирована комментарием: packaging PR идёт первым, Python upgrade после него
обновляется от свежего main. #195 не закрыт и не изменён.

## Следующий безопасный шаг

После green required CI и review merge packaging PR через GitHub. Затем
release-координатор использует только точный merged `origin/main` и выполняет
GCE runbook с image verifier, version/git_commit, lab synthetic feature smoke.

Не менять параллельно:

- `deploy/gcp-app/Dockerfile`;
- `deploy/gcp-app/deploy_to_gce.sh`;
- `deploy/gcp-app/verify_lab_assets.py`;
- `rag_server.py`, кроме штатного разрешения BUILD_VERSION.
