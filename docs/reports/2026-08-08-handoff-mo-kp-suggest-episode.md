# Handoff: МО KP suggest v4 + episode history

Дата: 2026-08-08

## Git

- repo: `akuazuk/protocol`
- branch: `cursor/mo-kp-suggest-dx-accuracy-agent1-pc1`
- worktree: `/private/tmp/protocol-task-mo-kp-suggest-dx-accuracy-pc1`
- base: `origin/main` at start of task

## Сделано

- Suggest v4: lexical + text→ICD bridge, clinical-only top, age audience
- `mo_dx_episode`: эпизод Dx из истории (не чужой прошлый Dx)
- Golden `tests/fixtures/mo_kp_suggest_golden.jsonl` (верно/неверно)
- API `/protocol-suggest` + review-pack тянут history bundle
- Рубрика МЗ plan учитывает clinical-hit КП
- `BUILD_VERSION`: `2026-08-08-104647Z-kp-suggest-episode`

## Не сделано

- Полная текстовая сверка плана с PDF КП
- Primary GCP smoke 3605554 после merge + `deploy_to_gce.sh`

## Тесты

```bash
pytest tests/test_mo_dx_episode.py tests/test_mo_kp_suggest_golden.py \
  tests/test_case_protocol_suggest.py tests/test_dx_query_expand.py \
  tests/test_mo_rubric_mz.py -q
```

## Следующая команда

После merge PR (не Render Action):

```bash
bash deploy/gcp-app/deploy_to_gce.sh
curl -fsS https://protocol.kravira.by/api/version
# smoke: case detail → protocol-suggest, плосковальгус / ортопедия
```

## Не трогать параллельно

`clinical_knowledge/case_protocol_suggest.py`, `protocol_match.py`, `dx_query_expand.py`, `mo_dx_episode.py`, golden fixture.
