# MO score calibration - frozen protocol

Status: C0-C4 completed on 2026-08-09. C5-C9 are not started.

## Scope

This experiment compares all existing MO scores with two shadow endpoints:

- Endpoint C: diagnosis-to-evidence concordance;
- Endpoint D: diagnosis-conditioned plan concordance, split into KP-grounded and
  lower-trust no-KP LLM routes.

It does not change production scoring, action queues, warehouse columns, or UI.

## Frozen pilot

- period: `2026-08-01..2026-08-08` (source data available through 2026-08-06);
- seed: `42`;
- target: 30 unique MO records;
- sentinel: visit alias `3643940`;
- maximum: 3 records per doctor;
- all available `training_use=1` review packs;
- minimum coverage exactly as registered in
  `docs/plans/2026-08-09-mo-score-ssot-llm-recompute-v3.md`.

The sampler uses `mis_id` as the unique record key and accepts visit/case IDs only
as aliases. This prevents a multi-record visit from joining to the wrong warehouse
row.

## Artifacts and PHI boundary

Secret artifacts remain on GCE under:

```text
/var/data/medical_exams/calibration/mo-score-v3-2026-08-01-2026-08-08/
```

The `secret/` directory contains identifiers, full clinical payloads, score
snapshots, replay details, and blind judge outputs. It is mode `0700`; files are
mode `0600`.

`public_manifest.json` and `smoke_summary.json` contain only aggregate counts and
hashes. No row identifiers, doctor labels, diagnoses, or clinical text are stored
in git.

## Reproducible command

Live Gemini calls are permitted only by the GCE wrapper:

```bash
bash deploy/gcp-llm/run_on_gce.sh \
  2026-08-01 2026-08-08 --calibration-smoke
```

The judge additionally refuses a live call unless both
`MO_LLM_EXECUTION_HOST=gce` and `RUN_HOST=gcp|gce` are set.

## C0-C4 gates

- C0: public audit must have `selected_n=30`, no deficits, sentinel present,
  every training-use record present, and doctor maximum at most 3.
- C1: every selected record must have a complete five-field replay comparison
  (overall plus four axes). Exact equality is measured, not silently assumed.
- C2: Endpoint C/D validators reject mixed routes, unsupported ICD mismatch,
  score-bearing blocked results, and no-KP claims of protocol compliance.
- C3: allowlisted prompt payloads must have zero forbidden engine fields and zero
  injected canary leakage.
- C4: five cases, two independent passes, both plan routes represented, all ten
  responses parsed, and zero leakage, geo, or model-call errors after retry.

## C0-C4 result

- sample: 30/30, all registered coverage constraints passed;
- current-engine replay: 30/30 audited, 0/30 exact across all five compared
  fields;
- GCE smoke: 5 cases x 2 passes, 10/10 parsed;
- routes: 4 KP-grounded outputs and 6 no-KP outputs;
- leakage / geo / runtime errors: 0 / 0 / 0;
- repeat agreement: Dx verdict 4/5, plan verdict 5/5;
- paired score differences: Dx mean 4.0 p.p. (max 20), plan mean 1.2 p.p.
  (max 5).

The replay drift is a finding, not a reason to overwrite the warehouse. C5 must
freeze the intended Arm D implementation/config hash and explain the old-vs-current
drift before candidate methodology is compared or production formulas change.
