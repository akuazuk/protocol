# Handoff: жёсткость оценок МО в Настройках

Дата: 2026-08-09

## Repo / deploy

- Feature PR: https://github.com/akuazuk/protocol/pull/117 (merged `ac76c086`)
- Polish branch: `cursor/mo-scoring-strictness-polish-pc1`
- Production GCE: `BUILD_VERSION=2026-08-09-192256Z-scoring-strictness-settings` (до polish-deploy)
- Plan: `docs/plans/2026-08-09-mo-scoring-strictness-settings-v1.md`

## Сделано

- Профиль `/var/data/medical_exams/config/mo_scoring_profile.json`
- GET/PUT `/api/methodist/mo/scoring-config`, POST `/api/methodist/mo/recompute`
- UI «Жёсткость оценок» + пояснения + poll статуса job
- Wire: zone bands / deep status / v4 risk caps / shadow / queue cutoff
- Next-load hook в night/inbound
- GCE smoke: save `strict` → recompute `2026-08-06` (447 cases, success) → restore `standard`

## Ограничения (осознанно)

- UI-пересчёт = витрина/зоны (`recompute_mo_days`), не полный deep/LLM rescore
- Risk-caps/status thresholds для уже записанного `overall` в cases.jsonl без deep-rescore не переписываются

## Тесты

```bash
pytest tests/test_mo_scoring_profile.py tests/test_mo_zone_scores.py -q
```

## Следующий шаг

Merge polish PR → `bash deploy/gcp-app/deploy_to_gce.sh` → открыть Настройки и проверить кнопки/poll.
