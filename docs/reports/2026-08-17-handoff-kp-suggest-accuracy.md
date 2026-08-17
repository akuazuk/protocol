# Handoff: точность КП suggest - прогон и golden 40

- repo: `akuazuk/protocol`
- branch (docs): `cursor/kp-suggest-eval-metrics-pc1`
- worktree: `/private/tmp/protocol-task-kp-suggest-eval-metrics-pc1`
- base / HEAD на момент старта: `origin/main` `b6c7381` (#155)
- план: `docs/plans/2026-08-14-mo-kp-suggest-accuracy-v2.md`
- primary: `https://protocol.kravira.by`

## Сделано

- Шаги 1-5, 7-9 плана в `main` (#149, #152, #154, #155).
- Golden 40: merge `#155` (`b6c7381`), `BUILD_VERSION` `2026-08-14-123546Z-kp-golden-40`.
- CSV-прогон GCE 26.07-13.08, без PHI в отчёте:
  `/var/data/medical_exams/reports/kp_suggest_eval_2026-07-26_08-13.json`
  - 7605 clinical, 71.2% available, 28.8% честное пусто
  - возраст решён 98.8%
  - `дет_нас` у взрослых: 6 (цель 0)
  - омнибус top-1: 568 (10.5% available); чаще ЛОР 2017
  - путь: diagnosis 7498, icd 64, complaints 43
  - 32395 с (~4.3 с/случай)
- Деплой `#155` на GCE после прогона (рестарт контейнера во время eval был запрещён).

## Не сделано

- Цель ≥75% hit не достигнута (71.2%).
- Омнибус ЛОР 2017 всё ещё часто top-1 (diag_overlap держит карточку).
- 6 взрослых с `дет_нас` top-1.

## Тесты

- Golden 40/40 локально; CI lint-and-test на `#155` зелёный.
- Baseline красный CI не открывали.

## Deploy / smoke

- merge SHA: `b6c7381` (#155)
- GCE deploy 2026-08-17: `deploy_to_gce.sh` ok
- `https://protocol.kravira.by/health/live` ok
- `/api/version` = `2026-08-14-123546Z-kp-golden-40` (`git_commit` null, версия совпала)

## Не трогать параллельно

- `clinical_knowledge/rceth_sync/` и план Rceth
- чужие open PR (#148 Gemini billed key и старше)

## Следующая команда

`curl -fsS https://protocol.kravira.by/api/version`
