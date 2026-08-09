# Handoff: MO score calibration C0-C4

Date: 2026-08-09

## Repository state

- repo: `akuazuk/protocol`
- branch: `cursor/mo-score-calibration-harness-pc1`
- worktree: `/private/tmp/protocol-task-mo-score-calibration-harness-pc1`
- base: `95e457b02e2b96d2167b87f8bc148c053874c308`
- implementation HEAD: `a7760ea8`
- PR: https://github.com/akuazuk/protocol/pull/108
- merge/deploy: not performed

## Completed

### C0 - sampler and PHI boundary

- Added a deterministic constrained sampler for the registered 30-case pilot.
- Joined L1 score rows to the full secure clinical CSV by canonical `mis_id`;
  visit/case IDs are aliases. This fixed a discovered multi-record visit join bug.
- Secret manifest, full case pack, engine snapshot, replay, and blind outputs stay
  on GCE under `/var/data/medical_exams/calibration/` with restricted permissions.
- The public manifest contains only aggregate coverage and artifact hashes.
- Final audit: 30 selected, no deficits, sentinel present, all 5 available
  `training_use=1`, maximum 3 records per doctor.
- Coverage: 13 specialties; all five score bands at least 4; ICD/Dx disputes 4;
  KP matched 8; KP unmatched 22; exam results 18; treatment 30.

Self-check: public manifest explicitly asserts no identifiers, doctor labels, or
clinical text. Unit tests inject secret canaries and verify they do not appear in
public output.

### C1 - all-score snapshot and replay

- Snapshot retains overall v4/v3, four axes, three zones, rubric, №55, findings,
  action membership, ICD pipeline, existing LLM output, and version/content hashes.
- Replayed the current v4 scorer on all 30 full payloads.
- Replay audit completed for 30/30 with five comparable fields per case and zero
  runtime errors.
- Exact reproducibility is 0/30. Aggregate differences were observed in overall
  and axes. This is a real drift finding; it was not hidden by overwriting data.

Self-check outcome: C1 procedure passed, equality did not. Before C5, freeze the
intended Arm D version/config and explain old snapshot versus current replay drift.
Do not change production formulas or recompute the warehouse from this result.

### C2 - Endpoint C/D contracts

- Endpoint C validates diagnosis evidence, evidence-slot provenance, blocked/NA
  semantics, ICD fit, and potential harm.
- A meaningful text diagnosis without ICD is eligible and receives no ICD-absence
  penalty.
- Endpoint D enforces exactly one route: KP-grounded with trust A/B and source
  references, or lower-trust `llm_no_kp` without protocol-compliance claims.
- Mixed `plan_protocol_pct` and `plan_general_llm_pct` outputs are rejected.

Self-check: synthetic tests cover good/poor/blocked/NA, unsupported ICD mismatch,
low-trust KP fallback, and no-KP protocol-claim rejection.

### C3 - blind prompts and leakage guards

- Stage C sees only diagnosis/ICD plus complaints, history, status, and exams.
- Stage D accepts the diagnosis as premise and does not receive Stage C verdicts.
- KP-grounded prompts receive only the selected protocol requirements/source refs;
  no-KP prompts do not receive matcher results.
- Engine scores, zones, №55, findings, action reasons, and queue state are denied by
  an allowlist plus automated forbidden-key/canary audit.
- Live calls are hard-blocked outside the GCE contour.

Self-check: local dry-run and leakage tests pass. No live Gemini call was made from
the Mac.

### C4 - GCE smoke

- GCE VM: `protocol-app`, `europe-central2-a`.
- Model: `gemini-3.6-flash`.
- Five cases, two independent passes: 10 outputs.
- Route coverage: 4 KP-grounded outputs, 6 no-KP outputs.
- Parse success: 10/10.
- Leakage / geo / runtime errors: 0 / 0 / 0.
- Paired agreement: Dx verdict 4/5; plan verdict 5/5.
- Mean absolute score difference: Dx 4.0 p.p.; plan 1.2 p.p.
- Maximum absolute score difference: Dx 20 p.p.; plan 5 p.p.

Self-check found and corrected three harness issues before the passing run:

1. L1 rows did not contain full clinical text; secure CSV join was added.
2. `visit_id` was not unique for MO records; canonical identity was changed to
   `mis_id` with visit aliases.
3. The first smoke selected only no-KP rows; route-balanced selection now requires
   both KP-grounded and no-KP paths.

## Verification

- `23 passed`:
  `tests/test_mo_score_calibration_sample.py`,
  `tests/test_mo_score_calibration_contracts.py`,
  `tests/test_mo_score_calibration_blind.py`.
- `40 passed` including regression suites
  `tests/test_mo_llm_action_judge.py` and `tests/test_case_protocol_suggest.py`.
- `bash -n deploy/gcp-llm/run_on_gce.sh`
- Python compile checks for all new modules/scripts.
- IDE lint: no diagnostics in changed Python/test files.
- Final GCE summary: `passed=true`, 10/10 parsed, both routes covered.

## Not completed

- C5 pilot 30 × 2 and LLM disagreement adjudication.
- C6 methodist labels.
- C7 candidate/ensemble metrics and bootstrap.
- C8 provisional methodology selection.
- C9 confirmatory cohort.
- Any production score, queue, warehouse, UI, merge, or deploy change.

## Production state

- production changed: no
- production SHA: unchanged/not inspected for this shadow experiment
- production smoke: not applicable
- secret data copied to git/local reports: no

## Safe next command

After this PR is green and merged, start C5 only after documenting which current
engine/config hash is the frozen Arm D reference:

```bash
scripts/ops/git_task_start.sh mo-score-calibration-pilot-c5 --pc=pc1 \
  --branch=cursor/mo-score-calibration-pilot-c5-pc1
```

## Files not to edit in parallel

- `scripts/build_mo_score_calibration_sample.py`
- `scripts/run_mo_calibration_blind_judge.py`
- `clinical_knowledge/mo_dx_evidence_score.py`
- `clinical_knowledge/mo_plan_protocol_score.py`
- `deploy/gcp-llm/run_on_gce.sh`
- `docs/plans/2026-08-09-mo-score-ssot-llm-recompute-v3.md`
