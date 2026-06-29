# Symptom ICD probe (100 queries)

- Generated: 2026-06-29 09:14 UTC
- Mode: local/no-gemini
- BUILD: `?`
- Probes: **100** (errors 0)

## ICD step-2 metrics

- Top-1 clinically plausible (prefix/heuristic): **100.0%**
- Top-3 plausible: **100.0%**
- Fully OK verdict: **100.0%**
- Bad prefix (T/X/Y/Z) in top-3: **0**
- Exotic A** fever in top-4 (URI/GI/ENT): **0**
- Empty ICD: **0**
- Top-1 miss (plausible in top-3): **0**
- Avg latency: **140** ms · p95 **185** ms

## Verdicts

- `ok`: 100

## By group

- **allergy** (1): top1 100.0% · fails: -
- **cardio** (10): top1 100.0% · fails: -
- **derm** (6): top1 100.0% · fails: -
- **emergency** (4): top1 100.0% · fails: -
- **endo** (6): top1 100.0% · fails: -
- **ent** (7): top1 100.0% · fails: -
- **gi** (12): top1 100.0% · fails: -
- **hematology** (1): top1 100.0% · fails: -
- **nephro** (3): top1 100.0% · fails: -
- **neuro** (8): top1 100.0% · fails: -
- **obgyn** (8): top1 100.0% · fails: -
- **oncology** (1): top1 100.0% · fails: -
- **ophth** (4): top1 100.0% · fails: -
- **ortho** (2): top1 100.0% · fails: -
- **psych** (5): top1 100.0% · fails: -
- **rheum** (5): top1 100.0% · fails: -
- **uri** (10): top1 100.0% · fails: -
- **uri_ped** (2): top1 100.0% · fails: -
- **uro** (5): top1 100.0% · fails: -

## Recurring bad codes

- `A25`: 2×
- `A28.1`: 2×
- `A39.5`: 1×
- `A88.1`: 1×
- `A81.1`: 1×
- `A39.1`: 1×
- `A48.3`: 1×
- `A39.3`: 1×
- `A74.0`: 1×

## Worst cases

| id | group | verdict | top-1 | top-3 | query |
|----|-------|---------|-------|-------|-------|
