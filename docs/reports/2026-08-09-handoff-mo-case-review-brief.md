# Handoff: case review brief (R1-R5)

Дата: 2026-08-09
Repo: `akuazuk/protocol`
Branch: `cursor/mo-case-review-brief-pc1`
Worktree: `/private/tmp/protocol-task-mo-case-review-brief-pc1`
Plan: `docs/plans/2026-08-09-mo-case-review-quality-parity-v1.md`

## Сделано

- R1: `mo_case_review_brief` + API `review_brief` + UI «Итог разбора» + prefill решения
- R2: `mo_clinical_gaps` (complaint/exam/dx/therapy/noise/plan) shadow findings
- R3: suggest `_rehab_or_noise_penalty` (детская БА выше rehab/adult)
- R4: `mo_case_narrative` opt-in (`MO_CASE_NARRATIVE=0` default) + UI «Черновик ИИ»
- R5: gold fixture `mo_1_test`, `scripts/eval_mo_case_review_brief.py`, catalog §Z.1
- `BUILD_VERSION`: `2026-08-09-061543Z-mo-case-review-brief`

## Не сделано

- R0 warehouse recompute `2026-08-06` на GCE (после merge + deploy)
- Опрос методиста на 20 кейсах
- Расширение gold до 15 МО

## Тесты

```bash
.venv/bin/python -m pytest \
  tests/test_mo_clinical_gaps.py \
  tests/test_mo_case_review_brief.py \
  tests/test_case_protocol_suggest_asthma_rank.py -q
# + scripts/eval_mo_case_review_brief.py → gap_recall 1.0
```

## Следующая команда

```bash
# после merge PR в origin/main:
bash deploy/gcp-app/deploy_to_gce.sh
# затем recompute дня эталона на GCE (name_match / zones chips)
```

## Не трогать параллельно

- `frontend/web/shared/mo-app.js`, `rag_server.py` case detail
- `clinical_knowledge/case_protocol_suggest.py`
- `clinical_knowledge/mo_case_review_brief.py` / `mo_clinical_gaps.py`
