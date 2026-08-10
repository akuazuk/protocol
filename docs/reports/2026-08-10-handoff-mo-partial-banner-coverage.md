# Handoff: partial banner coverage fix

## Repo

- branch: `cursor/mo-partial-banner-fix-pc1`
- worktree: `/private/tmp/protocol-task-mo-partial-banner-fix-pc1`
- base: `origin/main` @ `9b3ec891`
- HEAD: `7c19fff8`
- PR: https://github.com/akuazuk/protocol/pull/126
- `BUILD_VERSION`: `2026-08-10-135336Z-partial-banner-coverage`

## Done

1. Root cause: L1 `split_kz_rows` trusted stale CSV `document_kind=diagnostic` while
   completeness used live `clinical_visit` (consult+UZI) → false `scoring_coverage`.
2. Code: live classify for eligibility; `count_llm_queue_pending` clears spend/geo errors.
3. GCE ops (patched running container + data): resume-score 04..08, recompute 07/09.
   All days `partial=false`, coverage 100%. Aug 9 (UI «вчера» on 10.08) pending=0.

## Not done

- PR not merged; container has hot-patched modules until image redeploy from main.
- Soft advisory `llm_queue_pending` may still appear on 04/05/06/08 after rescoring
  refreshed the LLM queue with a few new unscored visit_ids (not the hard banner).
- Gemini spend cap still blocks successful LLM grades; raise cap then
  `deploy/gcp-llm/run_on_gce.sh` with `--resume --retry-errors` if needed.

## Tests

- GCE smoke: diagnostic→clinical_visit; spend-cap resolves pending.
- Unit tests in PR (run in CI): `test_mo_daily_pipeline`, `test_recompute_mo_days`.

## Deploy

Merge PR #126 → production release Action → confirm `/api/version` =
`2026-08-10-135336Z-partial-banner-coverage`. Smoke `/methodist/mo` «Вчера»: no hard
«Данные неполные (scoring_coverage, llm_queue_pending)».

## Next safe command

```bash
gh pr merge 126 --repo akuazuk/protocol --squash
# then release-coordinator watches Production Render / GCE release
```

## Hot files

- `scripts/run_mis_protocol_l1_batch.py` (`split_kz_rows`)
- `clinical_knowledge/mo_daily.py` (`count_llm_queue_pending`)
- `scripts/recompute_mo_days.py`, `clinical_knowledge/mo_orchestrator.py`
