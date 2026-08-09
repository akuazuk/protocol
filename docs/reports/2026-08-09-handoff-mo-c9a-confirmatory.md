# Handoff: C9A confirmatory proxy complete

Дата: 2026-08-09

## Repo

- branch: `cursor/mo-calibration-confirmatory-proxy-c9a-pc1`
- worktree: `/private/tmp/protocol-task-mo-calibration-confirmatory-proxy-c9a-pc1`
- PR: https://github.com/akuazuk/protocol/pull/113

## Сделано

- C6B/C7/C8B: LLM-proxy gold 22/22; C8B `no_stable_provisional` (tiny bad-n).
- C9A July `2026-07-26..2026-07-31`, n=100, seed 43:
  - blind flash: parse 99/100, leakage/geo 0;
  - proxy pro: parse 98/100, leakage/geo 0;
  - Dx labeled 82 / bad 13 → `provisional_shadow:blind.adjudicated_or_mean`
    (PR-AUC 0.81; with passes=1 equals pass_1/mean_2);
  - Plan labeled 75 / bad 26 → `provisional_shadow:ensemble.arm_d_blind_mean`
    (PR-AUC 0.90).
- PHI-safe: `eval/mo_score_calibration/c9a_public_summary.json`.
- `production_rollout.allowed=false`.

## Не сделано

- Production scoring / thresholds / queue / SSOT не менялись.
- Human methodist labels всё ещё 0/22.
- Shadow wiring в UI/action queue не включалась.

## Safe next

Owner review C9A provisional vs C8A/C8B:

1. Dx shadow: blind family (pass_1 / mean / adjudicated) - на confirmatory
   они совпадают при `passes=1`.
2. Plan shadow: `ensemble.arm_d_blind_mean` сильнее чистого blind на n=73.
3. Только после явного решения - отдельный PR на shadow signals
   (не SSOT recompute).

```bash
# sanity
python3 -c "import json; print(json.load(open('eval/mo_score_calibration/c9a_public_summary.json'))['provisional'])"
```

## Не трогать параллельно

- GCE `calibration/mo-score-v3-confirmatory-2026-07-26-2026-07-31/secret/`
- pilot C6 labels `methodist_labels.jsonl`
- production scoring paths
