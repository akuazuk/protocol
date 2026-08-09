# Handoff: MO score calibration C6 review pack

Date: 2026-08-09

## Repository state

- repo: `akuazuk/protocol`
- branch: `cursor/mo-score-calibration-methodist-c6-pc1`
- worktree: `/private/tmp/protocol-task-mo-score-calibration-methodist-c6-pc1`
- base: `e6b784323e5c7a954f2829d0fe1c77af27ec3857`
- HEAD/PR: pending
- merge/deploy: not performed

## Completed

- Built a blinded methodist pack from the frozen C5 sample and pilot.
- Selected every disagreement: 18 unique cases and 22 endpoint labels.
- Endpoint split: Dx 9, plan 13; 4 cases require both.
- Kept LLM passes and adjudication out of the review directory; comparison is
  created only after the human-label validator passes.
- Added strict validation for coverage, duplicates, verdict, score, harm,
  ICD fit, confidence, rationale, reviewer, and timestamp.
- Made pack creation idempotent: rerunning does not overwrite existing labels.
- Added the GCE-only `--calibration-methodist-pack` workflow.
- Added reviewer instructions in
  `eval/mo_score_calibration/methodist-labeling-guide.md`.

## GCE artifacts

Root:

```text
/var/data/medical_exams/calibration/mo-score-v3-2026-08-01-2026-08-08/
```

Secret artifact hashes:

- methodist cases:
  `d7b3b8a3ec10e57d545ac4ca6782662bb69df3a3c3638be2d0978748065a3fd1`;
- blank/current labels:
  `940fc97270821bdacbd7ac6664f019010becdc519f5564ef1cd3d88b5748892d`;
- frozen LLM comparison content hash:
  `ef10a1c358fd5ef8d4f0f2f4e2b5bcfc50f041107fffe6395b823c58aece78ce`.

PHI boundary:

- secret directory mode `0700`;
- secret files mode `0600`;
- public status contains no case IDs or clinical text;
- no secret artifact was copied to git or this report.

## Current C6 gate

- expected labels: 22;
- seen template rows: 22;
- complete labels: 0;
- unique cases: 18;
- missing/extra rows: 0/0;
- `passed=false`.

C6 is not complete. The pending step requires a real methodist. LLM
adjudication is intentionally not accepted as human gold.

## Verification

- C6 pack and blind-judge unit tests: 16 passed.
- Full calibration plus existing regression selection: 48 passed.
- Python compile and shell syntax passed.
- GCE pack build completed without a live LLM call.
- Generated public status matched 18 cases / 22 endpoints.
- IDE lint: no diagnostics.

## Production state

- scoring/action queue/warehouse/UI changed: no
- production deploy: not run
- `BUILD_VERSION`: `2026-08-09-141159Z-mo-calibration-c6-pack`

## Safe next command

After methodist labels are complete, validate them on GCE using the command in
`eval/mo_score_calibration/methodist-labeling-guide.md`. Do not start C7 until
`methodist_status.json` reports `passed=true`.

## Files not to edit in parallel

- `scripts/build_mo_calibration_methodist_pack.py`
- `deploy/gcp-llm/run_on_gce.sh`
- `eval/mo_score_calibration/methodist-labeling-guide.md`
- `docs/plans/2026-08-09-mo-score-ssot-llm-recompute-v3.md`
