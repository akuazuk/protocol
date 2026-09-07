# Handoff: P0 save lineage и допуск к обучению

Дата: 2026-09-07.
Repo: `akuazuk/protocol`.
Владелец: agent1 / pc1.
Branch: `cursor/mo-review-save-lineage-agent1-pc1`.
Worktree: `/private/tmp/protocol-task-mo-review-save-lineage-pc1`.
Base: `a6955ef2beac5decc48b338f3c0ae79d3479b232` (`origin/main`, PR #242).
`BUILD_VERSION`: `2026-09-07-101710Z-review-save-lineage`.
PR: открывается этой задачей. Merge и production на момент записи нет.

## Сверка перед работой

- `origin/main` = production `a6955ef2`, version `2026-09-07-085857Z-wave-e-acceptance`.
- `/health/live` на snapshot проверки был ok.
- Открытые PR: dependabot #194-#203, плюс #113 и #186. Их не мержить.
- `rag_server.py` формально занят #113/#186, но оба PR меняют его только как
  сопутствующую версию/старые docs. Конфликт одной `BUILD_VERSION` снимает
  rebase-скрипт. Жёсткого пересечения review-pack маршрута нет.
- `clinical_knowledge/mo_review_pack.py`, `mo_backend.py`, `mo-app.js` были свободны.
- Корневой checkout `/Users/pavelkuzauka/Cursor_Folders/Protocol` не менялся.

Предыдущие #217-#242 не повторялись. Этот этап закрывает только P0-1 и P0-2
из `CURSOR_REMAINING_WORK_2026-09-07.md`.

## Что сделано

Конкурентное сохранение пакета разбора:

- `expected_pack_id` и `expected_review_revision` в `save_review_pack`;
- запись pack + `crm_case_state` + событие в одной `BEGIN IMMEDIATE`;
- `crm_case_state` больше не `INSERT OR REPLACE` (последний writer не затирает
  чужую строку целиком без UPDATE существующей);
- idempotency key связан с canonical request hash: тот же ключ и тот же hash
  даёт replay, другой payload даёт `idempotency_payload_conflict`;
- client `evaluation_run_id` принимается только если совпадает с server run;
  пустой server run и любой client run отвергаются;
- `supersedes_pack_id` обязан принадлежать тому же `case_id`;
- HTTP 409 на все `SAVE_CONFLICT_ERRORS`;
- UI шлёт expected pack/revision, оставляет черновик при 409 и не предлагает
  молча перезаписать чужое решение.

Допуск к обучению:

- `training_use` больше не default true и не форсируется для `role=expert`;
- eligibility пишется в решение (`policy_version`, время, актор, отзыв);
- `revoke_training_eligibility` снимает допуск;
- `export_mo_review_gold.py` пропускает отозванные записи;
- чекбокс UI opt-in, без предварительной галочки.

Синтетическое воспроизведение: `tests/test_mo_review_pack_concurrency.py`.
Матрица остатка: `docs/reports/2026-09-07-mo-remaining-acceptance-matrix.md`.

## Проверки

```text
pytest tests/test_mo_review_pack_concurrency.py \
       tests/test_mo_review_pack.py \
       tests/test_mo_expert_auth.py \
       tests/test_mo_wave_e_acceptance.py::test_e20_review_pack_save_contract_exposes_all_guards
```

15 passed в concurrency-наборе плюс существующие review-pack/expert/E20.
`git diff --check` чистый. Ruff в среде агента не установлен.
Полный CI смотреть только на HEAD этой ветки после push.

## Что не сделано этим PR

- Compare-без-overwrite как отдельный экран двух пакетов: 409 оставляет
  черновик и просит открыть историю, но не рендерит side-by-side.
- HTTP revoke endpoint: функция есть, отдельный маршрут не добавлен.
- Массовая миграция старых `training_use=1` не выполнялась.
- Patient/time split и holdout не проверялись.
- History same-day, evaluated denominators, E04/E23 image, UX U01-U14 remainder.
- Merge, GCE deploy и production smoke.

Не писать «весь план CASE_REVIEW закрыт». Этот этап - только P0 save/training.

## Следующая безопасная команда

После merge этого PR, в новом clean worktree от свежего `origin/main`:

```text
scripts/ops/git_task_start.sh mo-history-same-day --pc=pc1 \
  --branch=cursor/mo-history-same-day-agent1-pc1
```

Не трогать параллельно: `clinical_knowledge/mo_review_pack.py` до merge;
не чинить корневой dirty checkout; не деплоить, пока HEAD не равен `origin/main`.
