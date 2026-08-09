# Handoff: жёсткость оценок МО в Настройках

Дата: 2026-08-09

## Repo

- Branch: `cursor/mo-scoring-strictness-settings-pc1`
- Worktree: `/private/tmp/protocol-task-mo-scoring-strictness-settings-pc1`
- Plan: `docs/plans/2026-08-09-mo-scoring-strictness-settings-v1.md`

## Сделано

- Профиль `/var/data/medical_exams/config/mo_scoring_profile.json` (soft/standard/strict/custom)
- GET/PUT `/api/methodist/mo/scoring-config`, POST `/api/methodist/mo/recompute`
- UI карточка «Жёсткость оценок» в Настройках
- Wire: zone bands, deep status, v4 risk caps, shadow cutoffs, queue attention cutoff
- Next-load hook: `scripts/mo_apply_scoring_profile_on_load.py` в night/inbound

## Не сделано / ограничения

- Пересчёт из UI - warehouse/zones (`recompute_mo_days`), не полный deep/LLM rescore
- Gemini spend cap может мешать night LLM, но не этому warehouse-пересчёту

## Тесты

```bash
pytest tests/test_mo_scoring_profile.py tests/test_mo_zone_scores.py -q
```

## Следующий шаг

PR → merge → `bash deploy/gcp-app/deploy_to_gce.sh` → smoke Настройки → сохранить пресет → пересчитать 1 день.
