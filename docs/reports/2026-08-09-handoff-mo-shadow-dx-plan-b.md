# Handoff: shadow Dx/Plan option B deployed

Дата: 2026-08-09

## Repo / deploy

- Feature PR: https://github.com/akuazuk/protocol/pull/114 (merged `89109b57`)
- Handoff PR: https://github.com/akuazuk/protocol/pull/115 (merged `55fe4a4e`)
- `origin/main`: `55fe4a4e`
- Production GCE `protocol.kravira.by`: `BUILD_VERSION=2026-08-09-185551Z-shadow-b-handoff`
- `/health/live` ok
- Render Action не используется (prod = GCE)

## Сделано

- Conservative shadow Dx/Plan (poor/critical only after soften)
- Case review UI block + queue filter/badge
- GCE runner + night hook in `mo_llm_range_runner.sh` (`MO_SHADOW_DX_PLAN=1` default)
- Official scores / SSOT / primary queue reason unchanged
- Deploy на GCE подтверждён через `/api/version`
- Loader smoke: fixture case `3646776` / `2026-08-06` → `available=true`, band `poor`

## GCE LLM smoke (блокер)

- `--date 2026-08-06 --limit 20` отработал пайплайн, но все 20 строк с
  `ResourceExhausted: 429 ... monthly spending cap` (оба `GOOGLE_API_KEY` и
  `GOOGLE_API_KEY_2`).
- Живые shadow-оценки недоступны, пока не поднимут spend cap в AI Studio
  (`https://ai.studio/spend`).
- После поднятия лимита: на GCE
  `python scripts/run_mo_shadow_dx_plan.py --date YYYY-MM-DD --resume`
  (resume уже переигрывает строки с `error`; или night `mo_llm_range_runner`).

## Дальше

1. Поднять Gemini spend cap (оба ключа).
2. Перегнать shadow с `--resume` за дни с cases (последний полный: `2026-08-06`).
3. В UI `/methodist/mo` проверить блок shadow на реальном кейсе.
4. При раздувании красных - ужесточить soften 45→40 в
   `clinical_knowledge/mo_shadow_dx_plan.py`.

## Не трогать параллельно

- Official scoring / recompute
- `methodist_labels.jsonl` pilot pack
