# Handoff: непрерывность эпизода и отбор на глубокий прогон

Дата: 2026-08-14

## Repo

- worktree: `/private/tmp/protocol-task-history-continuity-deep-pc1`
- branch: `cursor/history-continuity-deep-pc1`
- base: `origin/main` `afe0fdf`
- `BUILD_VERSION`: `2026-08-14-064130Z-history-continuity`

## Сделано

Слой A: если диагноз/план слабые, смотрим прошлые визиты к этому врачу / специальности.
Вердикт `known_episode` / `new_problem` / `no_history`. Официальный балл не меняем.
Очередь дня сортируется: safety → история может сменить вердикт → сильная модель.

## Не сделано

- extract января-25 июля и backfill `patient_key`
- слой B: чтение слотов prior + shadow rescore
- слой C: сильная модель на GCE
- merge / deploy

## Тесты

`tests/test_mo_history_continuity.py` и связанные MO frontend/yesterday.

## Следующая команда после merge

`GCE_OPS_USER=pavel SYNC_PROTOCOL_CORPUS=0 COPYFILE_DISABLE=1 bash deploy/gcp-app/deploy_to_gce.sh`
