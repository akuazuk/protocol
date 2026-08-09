# Handoff: shadow Dx/Plan option B deployed

Дата: 2026-08-09

## Repo / deploy

- PR: https://github.com/akuazuk/protocol/pull/114 (merged)
- `origin/main`: `89109b57`
- Production GCE `protocol.kravira.by`: `BUILD_VERSION=2026-08-09-184637Z-shadow-dx-plan-b`
- `/health/live` ok
- Render Action не используется (prod = GCE)

## Сделано

- Conservative shadow Dx/Plan (poor/critical only after soften)
- Case review UI block + queue filter/badge
- GCE runner + night hook in `mo_llm_range_runner.sh` (`MO_SHADOW_DX_PLAN=1` default)
- Official scores / SSOT / primary queue reason unchanged

## Дальше

1. Дождаться/проверить GCE smoke `--limit 20` за вчера.
2. Открыть разбор случая в `/methodist/mo` - блок shadow.
3. При раздувании красных - ужесточить soften 45→40 в
   `clinical_knowledge/mo_shadow_dx_plan.py`.
4. Полный night backfill пойдёт с обычным range-runner.

## Не трогать параллельно

- Official scoring / recompute
- `methodist_labels.jsonl` pilot pack
