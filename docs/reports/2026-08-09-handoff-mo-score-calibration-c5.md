# Handoff: MO score calibration C5

Date: 2026-08-09

## Repository state

- repo: `akuazuk/protocol`
- branch: `cursor/mo-score-calibration-pilot-c5-pc1`
- worktree: `/private/tmp/protocol-task-mo-score-calibration-pilot-c5-pc1`
- base: `4e7bd0df0164669fbeae070688e4b27affb5b32e`
- HEAD: pending final commit
- PR: pending publication
- merge/deploy: not performed

## Completed

### Frozen Arm D

- Added a combined fingerprint over v4 evaluation code, regulatory rules, config,
  1876 protocol-summary files, and relevant environment flags.
- Fingerprint:
  `9ab7bfcb5a84f47354aed8f916c08d5285319bd94d43bc2614a848cb54da2e49`.
- Blind judge/model/contracts fingerprint:
  `1e106ed8e0980127643aaf247f978a326a4d26f3bc167918274da843cdbefca0`.
- Frozen secret artifact hashes:
  - cases: `d4005d89aff9...bfac88f`;
  - manifest: `70b89aed8301...b2535a`;
  - engine snapshot: `111e7cea0f59...af62657`.
- Pilot resume now refuses changed secret files or a changed Arm D fingerprint.

### Replay drift diagnosis

- All 30 cases had complete five-field replay comparisons and no runtime error.
- Exact replay equality remained 0/30.
- Source `_content_hash` equalled warehouse `content_hash` in 0/30.
- Of 24 rows carrying `evaluation_v4`, source axes matched the warehouse snapshot
  in 23/24 for documentation/clinical/regulatory and 24/24 for safety, while
  source `evaluation_v4.score_pct` matched warehouse overall in only 6/24.
- Current replay versus source v4 differed in regulatory for 24/24 and in overall
  for 23/24.

Conclusion: the string `v4.0.0` is not sufficient provenance. Stored rows combine
different code/data/content states and cannot serve as the deterministic Arm D
baseline. The pilot uses current replay plus the full fingerprint; no warehouse
value was overwritten.

### Blind pilot and adjudication

- GCE only: `protocol-app`, `europe-central2-a`.
- Model: `gemini-3.6-flash`.
- 30 cases × 2 passes = 60/60 valid results.
- Routes: 19 KP-grounded cases, 11 no-KP cases.
- Errors: parse 0, leakage 0, geo 0, runtime 0.
- Disagreement endpoints: 22.
- Adjudication: 22/22 successful, Dx 9 and plan 13; leakage/errors 0.
- Checkpoint/resume writes every completed pass and deduplicates retries.

### Repeatability

- Dx verdict agreement: 28/30.
- Plan verdict agreement: 20/30.
- ICD-fit agreement: 27/30.
- Potential-harm agreement: Dx 30/30, plan 29/30.
- Dx score absolute difference: median 0, mean 9.04, max 99 p.p.
- Plan score absolute difference: median 0.5, mean 9.71, max 45 p.p.
- Three plan responses needed one schema-validation retry.

The max differences are too large to treat the LLM as gold. C6 methodist review is
required for every disagreement and the other pre-registered strata before C7.

## Self-checks and corrections

1. The first C5 attempt stopped before any model call because sampler coverage
   regressed by one `50-59` case.
2. Root cause: KP precomputation had replaced the full candidate pool with a
   bounded subset. Fixed by retaining every candidate and enriching only the KP
   subset.
3. The retry passed every sample constraint before model calls.
4. The GCE run completed and wrote a passing summary; only the local SSH wrapper
   remained stale. Container/host process inspection confirmed no active job
   before the local wrapper was stopped.
5. Pilot resume now reuses the frozen sample and validates all hashes instead of
   rebuilding a new sample.

## Verification

- 29 calibration tests passed.
- 46 tests passed with existing action-judge and protocol-suggest regressions.
- Python compile checks passed.
- `bash -n deploy/gcp-llm/run_on_gce.sh` passed.
- IDE lint: no diagnostics in changed Python/test files.
- GCE summary:
  - `result_n=60`;
  - `parse_success_n=60`;
  - `adjudication_n=22`;
  - `adjudication_success_n=22`;
  - `passed=true`.

## Not completed

- C6 methodist labels and adjudicated human gold.
- C7 candidate/ensemble metrics and bootstrap confidence intervals.
- C8 provisional methodology selection.
- C9 confirmatory cohort.
- Any production scoring, queue, warehouse, UI, or deploy change.

## Production state

- production changed: no
- production deploy: not run
- secret clinical artifacts copied to git/reports: no
- `BUILD_VERSION`: `2026-08-09-135354Z-mo-calibration-c5`

## Safe next command

After this branch is merged, create a separate C6 task to export a PHI-safe
methodist review index while clinical bodies remain in the secure GCE review pack:

```bash
scripts/ops/git_task_start.sh mo-score-calibration-methodist-c6 --pc=pc1 \
  --branch=cursor/mo-score-calibration-methodist-c6-pc1
```

## Files not to edit in parallel

- `scripts/build_mo_score_calibration_sample.py`
- `scripts/run_mo_calibration_blind_judge.py`
- `scripts/eval_mo_score_calibration.py`
- `deploy/gcp-llm/run_on_gce.sh`
- `docs/plans/2026-08-09-mo-score-ssot-llm-recompute-v3.md`
