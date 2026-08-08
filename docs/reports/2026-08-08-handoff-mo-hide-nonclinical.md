# Handoff: non-clinical outside tables/scores/KP/LLM

- repo: `akuazuk/protocol`
- branch: `cursor/mo-hide-nonclinical-pc1`
- worktree: `/private/tmp/protocol-task-mo-hide-nonclinical-pc1-pc1`
- base: `origin/main` @ `3d2bf745`
- HEAD: `11c33e34`
- PR: https://github.com/akuazuk/protocol/pull/71 (open, not merged)
- BUILD_VERSION: `2026-08-08-144502Z-hide-nonclinical`

## Done

- Case tables hard-gated to `clinical_visit` (API ignores opt-out / other kinds).
- Case detail / suggest: no №55, ICD, findings, rubric, KP, LLM for non-clinical.
- L1 split + warehouse soft-fill МКБ follow the same gate.
- UI toggle locked; URL `score_eligible_only=0` ignored.
- Plan: `docs/plans/2026-08-08-mo-nonclinical-exclude-hard-v1.md`

## Not done

- Merge PR #71
- Production Render release + GCE app deploy
- Smoke: table has no procedure/exam rows; direct non-clinical case URL shows text only

## Tests

- Passed: `test_mo_score_eligible_filter`, `test_mo_frontend_structure`, `test_mo_document_taxonomy`, `test_batch_document_kind_gates_over_mo_score_eligible_flag`

## Next safe command

```bash
gh pr merge 71 --repo akuazuk/protocol --squash
# then release-coordinator watches Production Render release + GCE deploy_to_gce.sh
```

## Do not touch in parallel

- `clinical_knowledge/mo_backend.py`, `mo_daily.py`, `mo_case_document.py`
- `rag_server.py` MO case-detail/suggest
- `frontend/web/shared/mo-app.js`, methodist HTML filters
